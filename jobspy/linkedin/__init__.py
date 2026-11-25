from __future__ import annotations

import random
import re
import time
from datetime import date
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from bs4.element import Tag
from curl_cffi import requests as curl_requests

from jobspy.linkedin.constant import (
    currencies,
    empty_page,
    headers,
    hourly_threshold,
    jobs_per_page,
    lower_limit,
    max_results,
    monthly_threshold,
    pay_intervals,
    salary_range,
    upper_limit,
)
from jobspy.linkedin.util import (
    criteria,
    is_job_remote,
    parse_location,
    retry_after,
)
from jobspy.model import (
    Compensation,
    CompensationInterval,
    Country,
    JobPost,
    Scraper,
    ScraperInput,
    annual_factor,
)
from jobspy.util import (
    TLSRotating,
    check_status,
    create_session,
    description_fields,
    get_enum_from_job_type,
    plain_converter,
)


class LinkedIn(Scraper):
    base_url = "https://www.linkedin.com"
    delay = (2.5, 4.5)
    """Seconds between any two requests, search pages and job pages alike"""
    retry_statuses = frozenset({429, 500, 502, 503, 504})
    max_retries = 4
    backoff = 5.0
    """The first wait after a 429, doubled for each retry after it"""
    max_wait = 120.0
    slowdown_limit = 4.0
    """Each 429 makes the pace 1.5 times slower, up to this; each success eases it back"""

    def scrape(self, scraper_input: ScraperInput) -> list[JobPost]:
        self.scraper_input = scraper_input
        # A browser's TLS fingerprint and kept cookies: LinkedIn throttles a client that
        # looks new on every request sooner than one that looks like a returning browser.
        self.session = create_session(
            TLSRotating(scraper_input.proxies), scraper_input, headers
        )
        self.next_request = 0.0
        self.slowdown = 1.0
        for name in ("is_remote", "job_type"):
            if getattr(scraper_input, name):
                self.warn(f"{name} isn't supported, ignoring it")
        params = self._search_params()
        job_list: list[JobPost] = []
        seen = set()
        start = scraper_input.offset
        page = 1

        while len(job_list) < scraper_input.results_wanted and start < max_results:
            self.log.info(f"search page: {page}")
            try:
                response = self._get(
                    f"{self.base_url}/jobs-guest/jobs/api/seeMoreJobPostings/search",
                    params={**params, "start": start},
                )
                check_status(response)
            except Exception as e:
                self.fail(str(e))
                break

            soup = BeautifulSoup(response.text, "html.parser")
            job_cards = soup.find_all(class_="base-search-card")
            if not job_cards:
                if response.text.strip() != empty_page:
                    self.fail("unexpected page with no jobs (blocked?)")
                elif start:
                    self.log.warning(
                        f"empty page at start={start} (the end of the results, or throttled)"
                    )
                break

            new_cards = []
            for job_card in job_cards:
                job_id = str(job_card.get("data-entity-urn", "")).split(":")[-1]
                if job_id and job_id not in seen:
                    seen.add(job_id)
                    new_cards.append((job_id, job_card))

            for job_id, job_card in new_cards:
                try:
                    job_post = self._process_job(job_card, job_id)
                except Exception as e:
                    self.log.warning(f"skipping job: {e}")
                    continue
                job_list.append(job_post)
                if len(job_list) >= scraper_input.results_wanted:
                    break

            if not new_cards or len(job_cards) < jobs_per_page:
                break
            start += jobs_per_page
            page += 1

        return job_list

    def _get(
        self, url: str, params: dict[str, str | int] | None = None
    ) -> curl_requests.Response:
        """
        A GET paced after the previous request; a 429 or a server error is retried after
        Retry-After, or after a backoff that doubles. LinkedIn's 429 is a short burst limit
        without a Retry-After, which a few seconds' wait clears.
        """
        for attempt in range(self.max_retries + 1):
            wait = self.next_request - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            if self.session.rotates:  # one visitor's cookies seen from many addresses
                self.session.cookies.clear()
            response = self.session.get(url, params=params, allow_redirects=True)
            status = response.status_code
            if status == 429:
                self.slowdown = min(self.slowdown * 1.5, self.slowdown_limit)
            elif status == 200:
                self.slowdown = max(self.slowdown * 0.9, 1.0)
            self.next_request = (
                time.monotonic() + random.uniform(*self.delay) * self.slowdown
            )
            if status not in self.retry_statuses or attempt == self.max_retries:
                return response
            pause = min(
                retry_after(response.headers.get("retry-after"))
                or self.backoff * 2**attempt * random.uniform(1, 1.5),
                self.max_wait,
            )
            self.log.info(f"status {status}, retry {attempt + 1} in {pause:.0f}s")
            self.next_request = time.monotonic() + pause
        raise AssertionError("unreachable")

    def _search_params(self) -> dict[str, str | int]:
        scraper_input = self.scraper_input
        params: dict[str, str | int | None] = {
            "keywords": scraper_input.search_term,
            "location": scraper_input.location,
            "distance": scraper_input.distance,
        }
        if scraper_input.easy_apply:
            params["f_AL"] = "true"
        if scraper_input.linkedin_company_ids:
            params["f_C"] = ",".join(map(str, scraper_input.linkedin_company_ids))
        if scraper_input.hours_old:
            params["f_TPR"] = f"r{scraper_input.hours_old * 3600}"
        return {name: value for name, value in params.items() if value is not None}

    def _process_job(self, job_card: Tag, job_id: str) -> JobPost:
        title_tag = job_card.find("h3", class_="base-search-card__title")
        if title_tag is None:
            raise ValueError("no title")
        title = title_tag.get_text(strip=True)

        company_tag = job_card.find("h4", class_="base-search-card__subtitle")
        if company_tag is None:
            raise ValueError("no company")
        company_link = company_tag.find("a", href=True)
        location_tag = job_card.find("span", class_="job-search-card__location")
        if location_tag is None:
            raise ValueError("no location")
        location = parse_location(location_tag.text.strip())
        time_tag = job_card.find("time")
        try:
            date_posted = (
                date.fromisoformat(str(time_tag["datetime"])) if time_tag else None
            )
        except (TypeError, KeyError, ValueError):
            date_posted = None
        job_details = (
            self._fetch_details(job_id) if self.scraper_input.fetch_description else {}
        )

        return JobPost(
            id=f"li-{job_id}",
            title=title,
            company=company_tag.get_text(strip=True),
            company_url=str(company_link["href"]).split("?")[0]
            if company_link
            else None,
            location=location,
            is_remote=is_job_remote(title, location),
            date_posted=date_posted,
            job_url=f"{self.base_url}/jobs/view/{job_id}",
        ).model_copy(update=job_details)

    @staticmethod
    def _parse_salary(text: str) -> Compensation:
        """ "$117,000.00/yr - $234,000.00/yr" """
        match = re.fullmatch(
            r"([^\d\s]+) ?([\d,]+\.\d\d)(/\w+) - [^\d\s]+ ?([\d,]+\.\d\d)/\w+",
            " ".join(text.split()),
        )
        if not match:
            return Compensation()
        currency, low, interval, high = match.groups()
        return Compensation(
            interval=pay_intervals.get(interval),
            min_amount=float(low.replace(",", "")),
            max_amount=float(high.replace(",", "")),
            currency=currencies.get(currency, currency),
        )

    @staticmethod
    def _guess_salary(description: str | None) -> Compensation:
        """The first "$min - $max" of the text; the interval from the size of the amounts"""
        match = salary_range.search(description or "")
        if not match:
            return Compensation()
        low, high = (float(match[i].replace(",", "")) for i in (1, 3))
        if match[2] or match[4]:
            low, high = (round(x * 1000, 2) if x < 1000 else x for x in (low, high))
        if low < 1:
            return Compensation()
        if low < hourly_threshold:
            interval, below = CompensationInterval.HOURLY, hourly_threshold
        elif low < monthly_threshold:
            interval, below = CompensationInterval.MONTHLY, monthly_threshold
        else:
            interval, below = CompensationInterval.YEARLY, upper_limit + 1
        factor = annual_factor[interval.value]
        if high >= below or not (
            lower_limit <= low * factor < high * factor <= upper_limit
        ):
            return Compensation()
        return Compensation(
            interval=interval, min_amount=low, max_amount=high, currency="USD"
        )

    def _fetch_details(self, job_id: str) -> dict[str, object]:
        """
        The job page's description, pay, logo and criteria; empty when the page can't be read
        """
        try:
            response = self._get(f"{self.base_url}/jobs/view/{job_id}")
            check_status(response)
        except Exception as e:
            self.warn(f"job {job_id}: {e}")
            return {}
        path = urlparse(response.url).path
        if not path.startswith("/jobs/view/"):
            self.warn(f"job {job_id}: redirected to {path}")
            return {}

        soup = BeautifulSoup(response.text, "html.parser")
        div_content = soup.find("div", class_="show-more-less-html__markup")
        if div_content:
            div_content.attrs.clear()
        html = div_content.prettify(formatter="html") if div_content else None
        fields = description_fields(html, self.scraper_input.description_format)
        logo_image = soup.find("img", class_="artdeco-entity-image")
        salary_tag = soup.find("div", class_="compensation__salary")
        employment_type = criteria(soup, "Employment type")
        job_type = (
            get_enum_from_job_type(employment_type.lower().replace("-", ""))
            if employment_type
            else None
        )
        job_level = criteria(soup, "Seniority level")
        pay = self._parse_salary(salary_tag.get_text() if salary_tag else "")
        source = None
        if pay == Compensation() and self.scraper_input.country_indeed == Country.USA:
            pay, source = self._guess_salary(plain_converter(html)), "description"
        return {
            **fields,
            **pay.model_dump(),
            "salary_source": source,
            "job_type": [job_type] if job_type else None,
            "job_level": job_level.lower() if job_level else None,
            "job_function": criteria(soup, "Job function"),
            "company_industry": criteria(soup, "Industries"),
            "company_logo": logo_image.get("data-delayed-url") if logo_image else None,
        }
