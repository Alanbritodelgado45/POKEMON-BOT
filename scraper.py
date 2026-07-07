"""Pokemon TCG Elite Trainer Box (ETB) market price tracker for Chilean stores.

Scrapes search/collection pages from 10 Chilean TCG retailers, extracts ETB
prices in CLP, computes the daily national market average and appends
everything to etb_market_data.csv. Each store (and each item) is isolated:
a failure in one never halts the rest of the pipeline.
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
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0",
]


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
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }


# Accessories and bulk lots that mention ETB but are not an ETB unit.
JUNK_PATTERN = re.compile(
    r"protector|sleeve|funda|playmat|carpeta|binder|dados|dice"
    r"|\bcase\b|\d+\s*unidades",
    re.IGNORECASE,
)


def is_etb(name: str) -> bool:
    """Keyword filter: keep only names that look like an Elite Trainer Box unit."""
    lowered = name.lower()
    if JUNK_PATTERN.search(lowered):
        return False
    if re.search(r"\betb\b", lowered):
        return True
    return "elite" in lowered and ("trainer" in lowered or "box" in lowered)


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


class BaseScraper(ABC):
    """Base scraper: fetching, UA rotation, retries, pagination and filtering.

    Subclasses only describe the store's search URL and HTML structure.
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

    @staticmethod
    def first(node, selectors: list[str]):
        """First element matching any selector in order, with non-empty text."""
        for selector in selectors:
            el = node.select_one(selector)
            if el is not None and el.get_text(strip=True):
                return el
        return None

    def fetch(self, url: str) -> BeautifulSoup | None:
        for attempt in range(1, self.retries_per_page + 1):
            try:
                response = self.session.get(
                    url, headers=build_headers(), timeout=self.request_timeout
                )
                if response.status_code == 404:
                    # Typical end-of-pagination signal (WooCommerce et al.).
                    return None
                if response.status_code in (403, 429):
                    # Blocked or throttled: retry with a different User-Agent.
                    self.log.warning(
                        "HTTP %d on %s (attempt %d), rotating User-Agent",
                        response.status_code, url, attempt,
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
        """Scrape all pages of the store's search results."""
        return self._scrape_pages(self.search_url)

    def _scrape_pages(self, url_for_page) -> list[dict]:
        """Paginate url_for_page(1..max_pages); per-item errors are logged and skipped."""
        today = date.today().isoformat()
        rows: list[dict] = []
        seen: set[tuple[str, int]] = set()

        for page in range(1, self.max_pages + 1):
            url = url_for_page(page)
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
                if page == 1:
                    # 200 OK but zero product cards: layout change or bot wall.
                    self.log.warning("No product cards found on %s", url)
                break  # No more results: end of pagination.

            found_on_page = 0
            for name_raw, price_raw in items:
                try:
                    name = " ".join(name_raw.split())
                    if not is_etb(name):
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


class FlatsomeWooScraper(BaseScraper):
    """WooCommerce stores on the Flatsome theme (El Panteon, Charizstore)."""

    BASE: str = ""

    def search_url(self, page: int) -> str:
        # /search?q= 404s on WooCommerce; the real endpoint is ?s=.
        if page == 1:
            return f"{self.BASE}/?s=elite+trainer+box&post_type=product"
        return f"{self.BASE}/page/{page}/?s=elite+trainer+box&post_type=product"

    def parse_items(self, soup: BeautifulSoup) -> list[tuple[str, str]]:
        items = []
        for card in soup.select("div.product-small.box"):
            title_el = self.first(card, [
                ".woocommerce-loop-product__title a",
                ".woocommerce-loop-product__title",
            ])
            if title_el is None:
                continue
            # Prefer the sale price inside <ins>; fall back to the first amount.
            price_el = self.first(card, [
                "ins .woocommerce-Price-amount",
                ".price .woocommerce-Price-amount",
            ])
            price_text = price_el.get_text(strip=True) if price_el else ""
            items.append((title_el.get_text(strip=True), price_text))
        return items


class ElPanteonScraper(FlatsomeWooScraper):
    store_name = "El Panteon"
    BASE = "https://www.elpanteon.cl"


class CharizstoreScraper(FlatsomeWooScraper):
    store_name = "Charizstore"
    BASE = "https://charizstore.cl"


class ShopifyDawnScraper(BaseScraper):
    """Shopify stores on the Dawn theme (AFK Store, Collector Center)."""

    BASE: str = ""

    def search_url(self, page: int) -> str:
        return f"{self.BASE}/search?type=product&q=elite+trainer+box&page={page}"

    def parse_items(self, soup: BeautifulSoup) -> list[tuple[str, str]]:
        items = []
        for card in soup.select("div.card-wrapper"):
            title_el = card.select_one(".card__heading a")
            if title_el is None:
                continue
            price_el = self.first(card, [
                ".price-item--sale",
                ".price-item--regular",
            ])
            price_text = price_el.get_text(strip=True) if price_el else ""
            items.append((title_el.get_text(strip=True), price_text))
        return items


class AFKStoreScraper(ShopifyDawnScraper):
    store_name = "AFK Store"
    BASE = "https://afkstore.cl"


class CollectorCenterScraper(ShopifyDawnScraper):
    store_name = "Collector Center"
    BASE = "https://collectorcenter.cl"


class PiedraBrujaScraper(BaseScraper):
    """Piedra Bruja (Shopify, custom Tailwind theme)."""

    store_name = "Piedra Bruja"
    BASE = "https://www.piedrabruja.cl"

    def search_url(self, page: int) -> str:
        return f"{self.BASE}/search?q=elite+trainer+box&type=product&page={page}"

    def parse_items(self, soup: BeautifulSoup) -> list[tuple[str, str]]:
        items = []
        for card in soup.select("div.product-card"):
            title_el = card.select_one(".product-card__title")
            if title_el is None:
                continue
            # In this theme .price__regular always holds the current price,
            # even inside .price--on-sale blocks (.price__sale is the old one).
            price_el = self.first(card, [".price__regular", ".price"])
            price_text = price_el.get_text(strip=True) if price_el else ""
            items.append((title_el.get_text(strip=True), price_text))
        return items


class TheWayScraper(BaseScraper):
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
            title_el = self.first(card, self.TITLE_SELECTORS)
            price_el = self.first(card, self.PRICE_SELECTORS)
            if title_el is None:
                continue
            price_text = price_el.get_text(strip=True) if price_el else ""
            items.append((title_el.get_text(strip=True), price_text))
        return items


class MiraxScraper(TheWayScraper):
    """Mirax Hobbies. Sits behind a Cloudflare JS challenge: plain HTTP
    clients are usually rejected with 403, which is logged and skipped."""

    store_name = "Mirax Hobbies"
    BASE = "https://www.mirax.cl"
    max_pages = 2

    def search_url(self, page: int) -> str:
        return f"{self.BASE}/search?q=elite+trainer+box&page={page}"


class MercadoLibreScraper(BaseScraper):
    """Mercado Libre Chile. Covers both the legacy ui-search and the newer
    poly-card result markup. Datacenter IPs may get a bot wall (403/challenge);
    that is logged and skipped."""

    store_name = "Mercado Libre"
    max_pages = 2  # 48+ results per page is plenty for ETBs.

    def search_url(self, page: int) -> str:
        base = "https://listado.mercadolibre.cl/elite-trainer-box-pokemon"
        if page == 1:
            return base
        return f"{base}_Desde_{(page - 1) * 48 + 1}"

    def parse_items(self, soup: BeautifulSoup) -> list[tuple[str, str]]:
        cards = soup.select("li.ui-search-layout__item") or soup.select("div.poly-card")
        items = []
        for card in cards:
            title_el = self.first(card, [
                "a.poly-component__title",
                "h3.poly-component__title-wrapper a",
                "h2.ui-search-item__title",
                "a.ui-search-link[title]",
            ])
            if title_el is None:
                continue
            # Current price first; never the struck-through previous price.
            price_el = self.first(card, [
                ".poly-price__current .andes-money-amount__fraction",
                ".ui-search-price__second-line .andes-money-amount__fraction",
                ".andes-money-amount:not(.andes-money-amount--previous) "
                ".andes-money-amount__fraction",
            ])
            price_text = price_el.get_text(strip=True) if price_el else ""
            items.append((title_el.get_text(strip=True), price_text))
        return items


class MoiiJuegosScraper(BaseScraper):
    """MoiiJuegos (Bsale platform). The /search route is rendered client-side,
    so we walk the server-rendered Pokemon collections instead and rely on
    the ETB keyword filter."""

    store_name = "MoiiJuegos"
    BASE = "https://moiijuegos.cl"
    COLLECTIONS = ["tcg-pokemon", "preventa-pokemon-tcg"]
    max_pages = 4  # Per collection.

    def search_url(self, page: int) -> str:  # pragma: no cover - unused
        return self._collection_url(self.COLLECTIONS[0], page)

    def _collection_url(self, collection: str, page: int) -> str:
        return f"{self.BASE}/collection/{collection}?order=id&way=DESC&limit=24&page={page}"

    def scrape(self) -> list[dict]:
        rows: list[dict] = []
        for collection in self.COLLECTIONS:
            try:
                rows.extend(
                    self._scrape_pages(lambda p, c=collection: self._collection_url(c, p))
                )
            except Exception:
                self.log.exception("Collection '%s' failed", collection)
        # Deduplicate across collections (same product can appear in both).
        unique: dict[tuple[str, int], dict] = {
            (row["item_name"].lower(), row["price_clp"]): row for row in rows
        }
        return list(unique.values())

    def parse_items(self, soup: BeautifulSoup) -> list[tuple[str, str]]:
        items = []
        for card in soup.select("div.bs-collection__product"):
            title_el = self.first(card, [
                ".bs-collection__product-title",
                "a.bs-collection__product-info[title]",
            ])
            if title_el is None:
                continue
            price_el = self.first(card, [
                ".bs-collection__product-final-price",
                ".bs-collection__product-price",
            ])
            price_text = price_el.get_text(strip=True) if price_el else ""
            items.append((title_el.get_text(strip=True), price_text))
        return items


class MagicsurScraper(BaseScraper):
    """Magicsur (PrestaShop)."""

    store_name = "Magicsur"
    BASE = "https://www.magicsur.cl"

    def search_url(self, page: int) -> str:
        return (
            f"{self.BASE}/index.php?controller=search"
            f"&search_query=elite+trainer+box&page={page}"
        )

    def parse_items(self, soup: BeautifulSoup) -> list[tuple[str, str]]:
        items = []
        for card in soup.select(".js-product-miniature"):
            title_el = self.first(card, [
                ".product-title a",
                ".product-title",
            ])
            if title_el is None:
                continue
            price_el = self.first(card, [
                ".product-price",
                ".product-price-and-shipping .price",
            ])
            price_text = price_el.get_text(strip=True) if price_el else ""
            items.append((title_el.get_text(strip=True), price_text))
        return items


ALL_SCRAPERS: list[type[BaseScraper]] = [
    MercadoLibreScraper,
    MiraxScraper,
    CharizstoreScraper,
    CollectorCenterScraper,
    MoiiJuegosScraper,
    ElPanteonScraper,
    AFKStoreScraper,
    TheWayScraper,
    PiedraBrujaScraper,
    MagicsurScraper,
]


def run_all_scrapers() -> list[dict]:
    rows: list[dict] = []
    for scraper_cls in ALL_SCRAPERS:
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
    logging.info(
        "Daily market average: $%s CLP over %d items", f"{daily_avg:,}", len(new_df)
    )
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
