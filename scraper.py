"""Pokemon TCG Elite Trainer Box (ETB) market price tracker for Chilean stores.

Scrapes search results from multiple Chilean TCG stores, extracts ETB
prices in CLP, computes the daily market average and appends everything
to etb_market_data.csv. Each store is isolated: a failure in one never
halts the others.
"""

import logging
import random
import re
import sys
import time
from abc import ABC, abstractmethod
from datetime import date

import pandas as pd
import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

CSV_PATH = "etb_market_data.csv"
CSV_COLUMNS = ["date", "store", "item_name", "price_clp"]
AVERAGE_STORE_LABEL = "MARKET_AVERAGE"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) "
    "Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

# Titles must look like an ETB; filters out sleeves/singles that match the search.
ETB_NAME_PATTERN = re.compile(r"elite\s+trainer\s+box|\betb\b", re.IGNORECASE)


def build_headers() -> dict:
    """Generic browser-like headers with a rotated User-Agent."""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "es-CL,es;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }


def clean_price(raw: str) -> int | None:
    """Parse a Chilean price string ('$69.990', 'A partir de $550.000') to int CLP."""
    if not raw:
        return None
    digits = re.sub(r"[^\d]", "", raw)
    if not digits:
        return None
    value = int(digits)
    # Discard placeholders and garbage values (free items, truncated parses).
    if value < 1000:
        return None
    return value


class StoreScraper(ABC):
    """Base scraper: fetching, UA rotation, retries, pagination and filtering.

    Subclasses only describe the store's HTML structure via parse_items().
    """

    store_name: str = "UnknownStore"
    max_pages: int = 5
    request_timeout: int = 25
    retries_per_page: int = 3

    def __init__(self) -> None:
        self.log = logging.getLogger(self.store_name)
        self.session = requests.Session()

    @abstractmethod
    def search_url(self, page: int) -> str:
        """Return the search-results URL for a given 1-indexed page."""

    @abstractmethod
    def parse_items(self, soup: BeautifulSoup) -> list[tuple[str, str]]:
        """Return (item_name, raw_price_text) tuples found on the page."""

    def fetch(self, url: str) -> BeautifulSoup | None:
        for attempt in range(1, self.retries_per_page + 1):
            try:
                response = self.session.get(
                    url, headers=build_headers(), timeout=self.request_timeout
                )
                if response.status_code == 404:
                    # WooCommerce returns 404 past the last page: end of results.
                    return None
                if response.status_code == 403:
                    # Blocked: retry with a different rotated User-Agent.
                    self.log.warning(
                        "HTTP 403 on %s (attempt %d), rotating User-Agent", url, attempt
                    )
                    time.sleep(2 * attempt)
                    continue
                response.raise_for_status()
                return BeautifulSoup(response.text, "lxml")
            except requests.RequestException as exc:
                self.log.warning("Request failed (attempt %d) %s: %s", attempt, url, exc)
                time.sleep(2 * attempt)
        return None

    def scrape(self) -> list[dict]:
        """Scrape all pages; per-item errors are logged and skipped."""
        today = date.today().isoformat()
        rows: list[dict] = []
        seen: set[tuple[str, int]] = set()

        for page in range(1, self.max_pages + 1):
            url = self.search_url(page)
            soup = self.fetch(url)
            if soup is None:
                if page == 1:
                    self.log.error("Could not fetch any results from %s", url)
                break

            try:
                items = self.parse_items(soup)
            except Exception:
                self.log.exception("Layout parsing failed on %s", url)
                break

            if not items:
                break  # No more results: end of pagination.

            found_on_page = 0
            for name_raw, price_raw in items:
                try:
                    name = " ".join(name_raw.split())
                    if not ETB_NAME_PATTERN.search(name):
                        continue
                    price = clean_price(price_raw)
                    if price is None:
                        continue
                    key = (name.lower(), price)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "date": today,
                            "store": self.store_name,
                            "item_name": name,
                            "price_clp": price,
                        }
                    )
                    found_on_page += 1
                except Exception:
                    self.log.exception("Skipping unparseable item on %s", url)

            self.log.info("Page %d: %d ETBs extracted", page, found_on_page)
            time.sleep(random.uniform(1.0, 2.5))  # Polite delay between pages.

        return rows


class ElPanteonScraper(StoreScraper):
    """El Panteon (WooCommerce / Flatsome theme)."""

    store_name = "El Panteon"
    BASE = "https://www.elpanteon.cl"

    def search_url(self, page: int) -> str:
        # /search?q= 404s on this WooCommerce site; the real endpoint is ?s=.
        if page == 1:
            return f"{self.BASE}/?s=elite+trainer+box&post_type=product"
        return f"{self.BASE}/page/{page}/?s=elite+trainer+box&post_type=product"

    def parse_items(self, soup: BeautifulSoup) -> list[tuple[str, str]]:
        items = []
        for card in soup.select("div.product-small.box"):
            title_el = card.select_one(".woocommerce-loop-product__title a") or \
                card.select_one(".woocommerce-loop-product__title")
            if title_el is None:
                continue
            # Prefer the sale price inside <ins>; fall back to the first amount.
            price_el = card.select_one("ins .woocommerce-Price-amount") or \
                card.select_one(".price .woocommerce-Price-amount")
            price_text = price_el.get_text(strip=True) if price_el else ""
            items.append((title_el.get_text(strip=True), price_text))
        return items


class AFKStoreScraper(StoreScraper):
    """AFK Store (Shopify / Dawn theme)."""

    store_name = "AFK Store"
    BASE = "https://afkstore.cl"

    def search_url(self, page: int) -> str:
        return f"{self.BASE}/search?type=product&q=elite+trainer+box&page={page}"

    def parse_items(self, soup: BeautifulSoup) -> list[tuple[str, str]]:
        items = []
        for card in soup.select("div.card-wrapper"):
            title_el = card.select_one(".card__heading a")
            if title_el is None:
                continue
            price_el = card.select_one(".price-item--sale")
            if price_el is None or not price_el.get_text(strip=True):
                price_el = card.select_one(".price-item--regular")
            price_text = price_el.get_text(strip=True) if price_el else ""
            items.append((title_el.get_text(strip=True), price_text))
        return items


class TheWayScraper(StoreScraper):
    """The Way TCG (Shopify). Selector chains cover the common Shopify themes."""

    store_name = "The Way"
    BASE = "https://thewaytcg.cl"

    CARD_SELECTORS = [
        "div.card-wrapper",
        "div.product-card",
        "div.product-item",
        "li.grid__item div.card",
        "div.grid-product",
    ]
    TITLE_SELECTORS = [
        ".card__heading a",
        ".product-card__title",
        ".product-item__title",
        ".grid-product__title",
        "a.full-unstyled-link",
    ]
    PRICE_SELECTORS = [
        ".price-item--sale",
        ".price-item--regular",
        ".product-card__price",
        ".product-item__price",
        ".grid-product__price",
        ".price",
    ]

    def search_url(self, page: int) -> str:
        return f"{self.BASE}/search?q=elite+trainer+box&type=product&page={page}"

    def parse_items(self, soup: BeautifulSoup) -> list[tuple[str, str]]:
        cards = []
        for selector in self.CARD_SELECTORS:
            cards = soup.select(selector)
            if cards:
                break
        items = []
        for card in cards:
            title_el = next(
                (el for sel in self.TITLE_SELECTORS if (el := card.select_one(sel))),
                None,
            )
            price_el = next(
                (
                    el
                    for sel in self.PRICE_SELECTORS
                    if (el := card.select_one(sel)) and el.get_text(strip=True)
                ),
                None,
            )
            if title_el is None:
                continue
            price_text = price_el.get_text(strip=True) if price_el else ""
            items.append((title_el.get_text(strip=True), price_text))
        return items


def run_all_scrapers() -> list[dict]:
    rows: list[dict] = []
    for scraper_cls in (ElPanteonScraper, AFKStoreScraper, TheWayScraper):
        scraper = scraper_cls()
        try:
            store_rows = scraper.scrape()
            logging.info("%s: %d ETBs total", scraper.store_name, len(store_rows))
            rows.extend(store_rows)
        except Exception:
            # One broken store must never halt the rest.
            logging.exception("%s: store scrape failed entirely", scraper.store_name)
    return rows


def persist(rows: list[dict]) -> None:
    today = date.today().isoformat()
    new_df = pd.DataFrame(rows, columns=CSV_COLUMNS)

    daily_avg = int(round(new_df["price_clp"].mean()))
    logging.info("Daily market average: $%s CLP over %d items", f"{daily_avg:,}", len(new_df))
    avg_row = pd.DataFrame(
        [
            {
                "date": today,
                "store": AVERAGE_STORE_LABEL,
                "item_name": "Daily Average ETB Price",
                "price_clp": daily_avg,
            }
        ],
        columns=CSV_COLUMNS,
    )

    try:
        history = pd.read_csv(CSV_PATH)
        # Idempotent re-runs: replace today's snapshot instead of duplicating it.
        history = history[history["date"] != today]
    except FileNotFoundError:
        history = pd.DataFrame(columns=CSV_COLUMNS)

    combined = pd.concat([history, new_df, avg_row], ignore_index=True)
    combined.to_csv(CSV_PATH, index=False)
    logging.info("Wrote %d rows to %s", len(combined), CSV_PATH)


def main() -> int:
    rows = run_all_scrapers()
    if not rows:
        logging.error("No ETB data extracted from any store; CSV left untouched")
        return 1
    persist(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
