"""Check rendered HTML targets, anchors, images, and search coverage."""

import argparse
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids = set()
        self.links = []
        self.images = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])
        if tag in ("img", "script", "iframe", "source") and "src" in attrs:
            self.links.append(attrs["src"])
        if tag == "link" and "href" in attrs:
            self.links.append(attrs["href"])
        if tag == "img":
            self.images.append(attrs)


def local_target(site, page, url):
    parts = urlsplit(url)
    if parts.scheme or parts.netloc:
        return None
    path = unquote(parts.path)
    if path.startswith("/crabbymetrics/"):
        target = site / path.removeprefix("/crabbymetrics/")
    elif path.startswith("/"):
        return None
    else:
        target = page.parent / path if path else page
    if target.is_dir():
        target /= "index.html"
    return target.resolve(), unquote(parts.fragment)


def check_site(site):
    site = site.resolve()
    pages = {}
    for path in site.rglob("*.html"):
        if any(part.startswith(".") or part in ("_freeze", "site_libs")
               for part in path.relative_to(site).parts):
            continue
        page = Page()
        page.feed(path.read_text())
        pages[path] = page
    errors = []
    for path, page in pages.items():
        for url in page.links:
            target = local_target(site, path, url)
            if target is None:
                continue
            destination, fragment = target
            if not destination.is_relative_to(site) or not destination.is_file():
                errors.append(f"{path.relative_to(site)}: missing target {url}")
            elif fragment and destination in pages and fragment not in pages[destination].ids:
                errors.append(f"{path.relative_to(site)}: missing anchor {url}")
        for img in page.images:
            if not img.get("src"):
                errors.append(f"{path.relative_to(site)}: image has no source")
    search_path = site / "search.json"
    if not search_path.is_file():
        errors.append("search.json is missing")
    else:
        search = json.loads(search_path.read_text())
        indexed = {urlsplit(row["href"]).path for row in search}
        for path in pages:
            name = path.relative_to(site).as_posix()
            if name not in indexed:
                errors.append(f"search.json: page not indexed: {name}")
        for row in search:
            target = local_target(site, site / "index.html", row["href"])
            if target and (not target[0].is_file() or
                           (target[1] and target[0] in pages and
                            target[1] not in pages[target[0]].ids)):
                errors.append(f"search.json: invalid target: {row['href']}")
    return {"pages": len(pages), "images": sum(len(p.images) for p in pages.values()),
            "errors": sorted(set(errors))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path)
    result = check_site(parser.parse_args().site)
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result["errors"]))
