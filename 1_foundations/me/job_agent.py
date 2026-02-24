#!/usr/bin/env python3
"""
Supermicro Job Application Agent

Reads applicant profile from summary.txt and job URLs from jobs.txt,
then uses Playwright + Claude to navigate, fill, and optionally submit
each application.

Usage:
    python job_agent.py            # Dry run — previews fields, does NOT submit
    python job_agent.py --submit   # Live mode — fills and submits after confirmation
"""

import asyncio
import json
import re
import sys
from pathlib import Path

import anthropic
import pypdf
import yaml
from dotenv import load_dotenv
from playwright.async_api import BrowserContext, Page, async_playwright

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PROFILE_PATH = BASE_DIR / "summary.txt"
JOBS_PATH = BASE_DIR / "jobs.txt"
RESUME_PATH = BASE_DIR / "linkedin.pdf"

DRY_RUN = "--submit" not in sys.argv


def _escape_css_id(el_id: str) -> str:
    """Escape characters that are special in CSS selectors when present in HTML IDs.

    SuccessFactors uses IDs like '768:_input' — the colon is valid HTML but
    breaks CSS selector syntax unless escaped as '\\:'.
    """
    for ch in (":", ".", "[", "]", "(", ")"):
        el_id = el_id.replace(ch, "\\" + ch)
    return el_id


def _clean_label(label: str) -> str:
    """Normalize a field label for robust get_by_label matching.

    Strips SuccessFactors required markers (*), newlines, colons, and
    '(Required)' annotations so Playwright can find the element by its
    visible human-readable name.
    """
    # Remove leading * required marker (may be followed by newline/spaces)
    label = re.sub(r"^\*\s*", "", label.strip())
    # Normalize internal whitespace / newlines to single spaces
    label = " ".join(label.split())
    # Remove trailing colon
    label = label.rstrip(":").strip()
    # Remove trailing (Required) annotation
    label = re.sub(r"\s*\(Required\)\s*$", "", label).strip()
    return label


# ── Data loaders ─────────────────────────────────────────────────────────────

def load_profile() -> dict:
    with open(PROFILE_PATH) as f:
        return yaml.safe_load(f)


def load_resume_pdf() -> str:
    reader = pypdf.PdfReader(RESUME_PATH)
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def load_jobs() -> list[dict]:
    """Return list of {url, status} dicts parsed from jobs.txt."""
    jobs = []
    with open(JOBS_PATH) as f:
        for line in f:
            m = re.search(r'(https://jobs\.supermicro\.com/\S+)\s*\|\s*(\d)', line)
            if m:
                jobs.append({"url": m.group(1).rstrip(), "status": int(m.group(2))})
            else:
                # Legacy lines without status field — treat as not applied
                m2 = re.search(r'https://jobs\.supermicro\.com/\S+', line)
                if m2:
                    jobs.append({"url": m2.group(0).rstrip(), "status": 0})
    return jobs


def update_job_status(url: str, status: int) -> None:
    """Update the status field for the given URL in jobs.txt (0=not applied, 1=applied, 2=applying)."""
    with open(JOBS_PATH) as f:
        lines = f.readlines()
    with open(JOBS_PATH, "w") as f:
        for line in lines:
            if url in line:
                # Replace existing status or append it
                if re.search(r'\|\s*\d', line):
                    line = re.sub(r'\|\s*\d', f'| {status}', line)
                else:
                    line = line.rstrip() + f' | {status}\n'
            f.write(line)


def build_applicant_context(profile: dict) -> str:
    """Flatten relevant profile sections into a concise string for Claude."""
    public = profile.get("sharing_contexts", {}).get("public", {}).get("data", {})
    employers = profile.get("sharing_contexts", {}).get("employers", {}).get("data", {})
    colleagues = profile.get("sharing_contexts", {}).get("work_colleagues", {}).get("data", {})
    job_app = (profile.get("job_applications") or [{}])[0]
    app_data = job_app.get("application_data", {})
    demo = job_app.get("demographics", {})

    # Build work history block
    work_history_lines = []
    for job in employers.get("work_experience", []):
        line = f"  - {job.get('role')} at {job.get('company')}"
        if job.get("status") == "Current":
            line += " (Current)"
        achievements = job.get("achievements", [])
        if achievements:
            line += ": " + "; ".join(achievements)
        work_history_lines.append(line)
    work_history = "\n".join(work_history_lines) if work_history_lines else "  (none listed)"

    # Build skills block
    tech = colleagues.get("technical_skills", {})
    skills_lines = []
    for category, items in tech.items():
        if isinstance(items, list):
            skills_lines.append(f"  {category}: {', '.join(items)}")
    skills = "\n".join(skills_lines) if skills_lines else "  (none listed)"

    # Build notable projects block
    projects_lines = []
    for p in employers.get("notable_projects", []):
        projects_lines.append(f"  - {p.get('project')}: {p.get('description')}")
    projects = "\n".join(projects_lines) if projects_lines else "  (none listed)"

    # Leadership
    leadership = "\n".join(
        f"  - {l}" for l in employers.get("leadership_experience", [])
    ) or "  (none listed)"

    return f"""
Full Name:                  {public.get('name')}
Preferred Name:             {public.get('preferred_name')}
Professional Title:         {employers.get('professional_summary', {}).get('title')}
Years of Experience:        {employers.get('professional_summary', {}).get('years_experience')}
Location:                   {public.get('location')}
Salary Expectation:         {app_data.get('salary_expectation', 175000)}
Age 18+:                    {app_data.get('age_eligible', True)}
US Citizen or National:     {app_data.get('us_citizen_or_national', True)}
Requires Visa Sponsorship:  {app_data.get('requires_visa_sponsorship', False)}
Previous Supermicro Employee: {app_data.get('previous_supermicro_employee', False)}
Supermicro Relatives:       {app_data.get('supermicro_relatives', False)}
Referral Source:            {app_data.get('referral_source')}
Referring Employee:         {app_data.get('referring_employee')}
Prior Termination:          {app_data.get('previous_termination', False)}
Signature:                  {app_data.get('signature')}
Gender:                     {demo.get('gender')}
Ethnicity:                  {demo.get('ethnicity')}
Veteran Status:             {demo.get('veteran_status')}
Disability Status:          {demo.get('disability_status')}

Work History:
{work_history}

Technical Skills:
{skills}

Notable Projects:
{projects}

Leadership Experience:
{leadership}
""".strip()


# ── Browser session keepalive ─────────────────────────────────────────────────

async def _browser_keepalive(page: Page, stop_event: asyncio.Event) -> None:
    """Scroll and nudge the mouse every few seconds to prevent SuccessFactors
    from destroying the form iframe while the Claude API call is in flight."""
    scroll_down = True
    while not stop_event.is_set():
        try:
            await page.mouse.move(600, 400)
            await page.evaluate(f"window.scrollBy(0, {10 if scroll_down else -10})")
            scroll_down = not scroll_down
        except Exception:
            pass
        # Sleep in 1-second increments so we can react to stop quickly
        for _ in range(4):
            if stop_event.is_set():
                return
            await asyncio.sleep(1)


# ── Claude-powered form filler ────────────────────────────────────────────────

async def scrape_job_info(page: Page) -> str:
    """Extract job title and description text from the listing page."""
    title = ""
    description = ""

    # Job title — try common patterns
    for sel in ("h1", ".job-title", "[class*='jobTitle']", "[class*='job-title']"):
        try:
            el = page.locator(sel).first
            if await el.count() > 0:
                title = (await el.inner_text()).strip()
                break
        except Exception:
            pass

    # Job description body text
    for sel in (
        ".job-description",
        "[class*='jobDescription']",
        "[class*='job-description']",
        "[class*='description']",
        "article",
        "main",
    ):
        try:
            el = page.locator(sel).first
            if await el.count() > 0:
                description = (await el.inner_text()).strip()
                break
        except Exception:
            pass

    # Fall back to page title tag if nothing else worked
    if not title:
        title = await page.title()

    return f"Job Title: {title}\n\nJob Description:\n{description[:4000]}"


async def click_expand_all_link(page: Page) -> None:
    """
    Always explicitly expands the 'Job-Specific Information'
    section to guarantee its fields are visible.
    """
    targets = [page] + list(page.frames[1:])

    # 1. Try the explicit "Expand all sections" link first (exact or partial text)
    for target in targets:
        for locator in (
            target.get_by_role("link", name="Expand all sections"),
            target.get_by_role("button", name="Expand all sections"),
            target.locator("a:has-text('Expand all sections')"),
            target.locator("button:has-text('Expand all sections')"),
            target.locator("[class*='expand']:has-text('Expand')"),
        ):
            try:
                if await locator.count() > 0 and await locator.first.is_visible():
                    await locator.first.click()
                    print("  → Clicked 'Expand all sections'")
                    await page.wait_for_timeout(1_000)
                    # Still fall through to expand Job-Specific section explicitly
                    break
            except Exception:
                pass
        else:
            continue
        break
    else:
        # 2. Fallback: click every collapsed toggle individually
        print("  → 'Expand all sections' link not found — expanding toggles individually")
        COLLAPSED_SELECTORS = [
            '[aria-expanded="false"]',
            'details:not([open]) > summary',
            '.accordion-toggle.collapsed',
            '[data-toggle="collapse"].collapsed',
            '[data-bs-toggle="collapse"].collapsed',
            '[class*="section"][class*="collapsed"]',
            '[class*="panel"][class*="collapsed"]',
        ]
        for target in targets:
            for sel in COLLAPSED_SELECTORS:
                try:
                    for el in await target.query_selector_all(sel):
                        try:
                            if await el.is_visible():
                                await el.click()
                                await page.wait_for_timeout(200)
                        except Exception:
                            pass
                except Exception:
                    pass

    # 3. Explicitly expand the 'Job-Specific Information' section if still collapsed
    await _expand_named_section(page, "Job-Specific Information")
    await page.wait_for_timeout(600)


async def _handle_login_wall(page: Page) -> None:
    """
    Detect a login/sign-in page and pause so the user can log in manually
    before automation continues.  Checks both the URL and the DOM (password
    field + sign-in button) to be robust across SSO redirects.
    """
    url_lower = page.url.lower()
    login_url_hints = ("login", "signin", "sign_in", "logon", "auth", "/login")

    has_password = False
    has_signin   = False
    try:
        has_password = await page.locator('input[type="password"]').count() > 0
        has_signin   = await page.locator(
            "button:has-text('Sign In'), a:has-text('Sign In'), "
            "button:has-text('Log In'), input[type='submit']"
        ).count() > 0
    except Exception:
        pass

    is_login_page = (
        any(h in url_lower for h in login_url_hints)
        or (has_password and has_signin)
    )

    if is_login_page:
        print(
            "  → Login wall detected (SuccessFactors or SSO). "
            "Please log in via the browser."
        )
        input("  Press Enter after you have logged in and the application page is visible...")
        await page.wait_for_timeout(2_000)


async def _expand_named_section(page: Page, section_name: str) -> None:
    """Click a named section header/toggle if it is collapsed or closed."""
    targets = [page] + list(page.frames[1:])
    HEADER_TAGS = ["h2", "h3", "h4", "h5", "button", "a", "div", "span", "legend"]
    for target in targets:
        for tag in HEADER_TAGS:
            try:
                els = await target.query_selector_all(
                    f'{tag}:has-text("{section_name}")'
                )
                for el in els:
                    if not await el.is_visible():
                        continue
                    # Click if the element itself or its parent signals collapsed state
                    aria = await el.get_attribute("aria-expanded")
                    parent = await el.evaluate_handle("e => e.parentElement")
                    parent_aria = await parent.get_property("ariaExpanded") if parent else None
                    parent_aria_val = await parent_aria.json_value() if parent_aria else None

                    is_collapsed = (
                        aria == "false"
                        or parent_aria_val == "false"
                        or "collapsed" in (await el.get_attribute("class") or "")
                    )
                    # Always click — if already open this is harmless on most UIs;
                    # limit to collapsed/unknown to avoid toggling closed
                    if is_collapsed or aria is None:
                        await el.click()
                        print(f"  → Expanded section '{section_name}'")
                        await page.wait_for_timeout(500)
                        return
            except Exception:
                pass


async def discover_form_fields(page: Page) -> list[dict]:
    """
    Use Playwright to enumerate every visible, enabled form element across the
    main page and all iframes. Returns a list of field descriptors with selector,
    human-readable label, type, and (for <select>) available options.
    """
    FIELD_QUERY = (
        'input:not([type="hidden"]):not([type="file"])'
        ':not([type="submit"]):not([type="button"]):not([type="reset"]),'
        'select, textarea, [contenteditable="true"],'
        '[role="combobox"], [role="radiogroup"], [role="listbox"]'
    )
    all_fields = []
    targets = [page] + list(page.frames[1:])

    for target in targets:
        # Resolve URL for this target (Page has .url; Frame also has .url)
        try:
            target_url: str = target.url
        except Exception:
            target_url = ""

        try:
            elements = await target.query_selector_all(FIELD_QUERY)
        except Exception:
            continue

        for el in elements:
            try:
                if not await el.is_visible() or not await el.is_enabled():
                    continue

                tag      = await el.evaluate("e => e.tagName.toLowerCase()")
                is_ce    = await el.get_attribute("contenteditable") == "true"
                role     = await el.get_attribute("role") or ""
                el_type  = "contenteditable" if is_ce else role or (await el.get_attribute("type") or tag)
                el_id    = await el.get_attribute("id") or ""
                el_name  = await el.get_attribute("name") or ""
                ph       = await el.get_attribute("placeholder") or ""
                aria_lbl = await el.get_attribute("aria-label") or ""

                # Resolve label text via <label for="..."> then aria-labelledby
                label_text = ""
                if el_id:
                    try:
                        lbl = await target.query_selector(f'label[for="{el_id}"]')
                        if lbl:
                            label_text = (await lbl.inner_text()).strip()
                    except Exception:
                        pass
                if not label_text:
                    try:
                        labelledby = await el.get_attribute("aria-labelledby")
                        if labelledby:
                            lbl = await target.query_selector(f'#{labelledby}')
                            if lbl:
                                label_text = (await lbl.inner_text()).strip()
                    except Exception:
                        pass

                # Collect options for <select> and ARIA combobox/radiogroup/listbox
                options: list[str] = []
                if tag == "select":
                    try:
                        for opt in await el.query_selector_all("option"):
                            txt = (await opt.inner_text()).strip()
                            if txt:
                                options.append(txt)
                    except Exception:
                        pass
                elif role in ("combobox", "radiogroup", "listbox"):
                    try:
                        for opt in await el.query_selector_all(
                            "[role='option'], [role='radio'], li"
                        ):
                            txt = (await opt.inner_text()).strip()
                            if txt and len(txt) < 120:
                                options.append(txt)
                    except Exception:
                        pass

                # Build the most stable CSS selector: prefer id, then name
                if el_id:
                    sel = f"#{_escape_css_id(el_id)}"
                elif el_name:
                    sel = f'[name="{el_name}"]'
                else:
                    sel = await el.evaluate("""e => {
                        const tag = e.tagName.toLowerCase();
                        const type = e.type ? `[type="${e.type}"]` : '';
                        const cls = [...e.classList].slice(0, 2).join('.');
                        return cls ? `${tag}${type}.${cls}` : `${tag}${type}`;
                    }""")

                entry: dict = {
                    "selector": sel,
                    "label": label_text or aria_lbl or ph or el_name or el_id or f"{tag}[{el_type}]",
                    "type": el_type,
                    "frame_url": target_url,
                }
                if options:
                    entry["options"] = options
                all_fields.append(entry)

            except Exception:
                continue

    return all_fields


async def fill_in_frame(
    page: Page,
    selector: str,
    action: str,
    value: str,
    label: str = "",
    frame_url: str = "",
) -> None:
    """Locate a form field and perform the requested action.

    Resolution order per frame:
      1. get_by_label with cleaned label text (immune to SPA ID regeneration)
      2. get_by_label with the raw label as fallback
      3. CSS selector (original ID-based, may be stale after re-render)

    When frame_url is provided the frame that originally hosted the field is
    tried first, which avoids wasting time on unrelated frames.
    """
    all_targets = [page] + list(page.frames[1:])

    # Put the frame that owned this field at the front
    if frame_url:
        preferred = [t for t in all_targets if getattr(t, "url", "") == frame_url]
        others    = [t for t in all_targets if getattr(t, "url", "") != frame_url]
        targets   = preferred + others
    else:
        targets = all_targets

    clean_lbl = _clean_label(label) if label else ""

    for target in targets:
        el = None

        # 1. get_by_label with cleaned text — most stable across re-renders
        if clean_lbl:
            try:
                loc = target.get_by_label(clean_lbl, exact=False).first
                if await loc.count() > 0:
                    el = loc
            except Exception:
                pass

        # 2. get_by_label with raw label (in case stripping changed meaning)
        if el is None and label and label != clean_lbl:
            try:
                loc = target.get_by_label(label, exact=False).first
                if await loc.count() > 0:
                    el = loc
            except Exception:
                pass

        # 3. CSS selector (original ID — may be stale after SPA re-render)
        if el is None:
            try:
                loc = target.locator(selector).first
                if await loc.count() > 0:
                    el = loc
            except Exception:
                pass

        if el is None:
            continue

        try:
            if action == "fill":
                # contenteditable divs don't support .fill(); clear then type
                ce = await el.get_attribute("contenteditable")
                if ce == "true":
                    await el.click()
                    await el.evaluate("e => e.innerHTML = ''")
                    await el.type(value)
                else:
                    await el.fill(value)
            elif action == "select_option":
                await el.select_option(label=value)
            elif action == "check":
                if value.lower() in ("true", "yes", "1"):
                    await el.check()
                else:
                    await el.uncheck()
            elif action == "click_option":
                # Custom ARIA dropdown / radio group: click the trigger to open it,
                # then find and click the option whose text matches value.
                await el.click()
                await page.wait_for_timeout(500)
                found = False
                for opt_sel in (
                    "[role='option']",
                    "[role='radio']",
                    "[role='menuitem']",
                    "li",
                    "[class*='option']",
                ):
                    try:
                        opt = target.locator(opt_sel).filter(has_text=value).first
                        if await opt.count() > 0 and await opt.is_visible():
                            await opt.click()
                            found = True
                            break
                    except Exception:
                        continue
                if not found:
                    raise RuntimeError(f"Option '{value}' not found after opening widget")
            await page.wait_for_timeout(250)
            return
        except Exception:
            continue

    raise RuntimeError(f"Selector not found in any frame: {selector}")


async def analyze_and_fill(
    page: Page,
    client: anthropic.Anthropic,
    applicant_context: str,
    job_context: str = "",
) -> None:
    """
    Discover all visible form fields via Playwright DOM querying, send the
    structured field list to Claude, and execute the returned fill instructions.

    The Claude API call runs in a thread-pool executor while a keepalive
    coroutine scrolls the page, preventing SuccessFactors from destroying the
    form iframe due to inactivity during the (potentially 10-30 second) API
    round-trip.
    """
    fields = await discover_form_fields(page)

    if not fields:
        print("  ⚠  No visible form fields found on this step — skipping")
        return

    print(f"  → Discovered {len(fields)} field(s) on page")

    # Build selector → frame_url map so fill_in_frame can route to the right frame
    selector_to_frame_url: dict[str, str] = {
        f["selector"]: f.get("frame_url", "") for f in fields
    }

    # Strip frame_url before sending to Claude — it's routing metadata, not form data
    fields_for_claude = [
        {k: v for k, v in f.items() if k != "frame_url"} for f in fields
    ]
    fields_json = json.dumps(fields_for_claude, indent=2)
    job_section = f"\nJob being applied for:\n{job_context}\n" if job_context else ""

    prompt = f"""You are an AI agent filling out a job application form on behalf of an applicant.

Applicant information:
{applicant_context}
{job_section}
Form fields discovered on the current page (each has selector, label, type, and optional options):
{fields_json}

Map the applicant's data to these fields and return ONLY a JSON array of fill instructions.
Each instruction must have exactly these keys:
  "label"    – the field's label (for logging)
  "selector" – copied exactly from the field list above
  "value"    – value to enter, as a string
  "action"   – one of: "fill" (text/textarea/number/contenteditable), "select_option" (select), "check" (checkbox/radio), "click_option" (combobox/radiogroup/listbox — custom ARIA dropdowns)

Rules:
- Include ALL fields — do NOT skip any visible field, especially those in the
  "Job-Specific Information" section.
- For select_option, value must exactly match one of the field's listed options.
- For No fields use "false", "no", or "0".
- For Yes fields use "true", "yes", or "1".
- For every open-text or contenteditable field (cover letters, "why interested",
  qualifications, relevant experience, additional information, etc.) write a
  compelling 3-5 sentence response tailored to the job title and description,
  drawing from the applicant's work history, skills, projects, and resume.
  Never leave these blank.
- For fields where applicant data is genuinely absent (e.g. a phone number not
  in the profile), omit only those fields.
- For fields with type "combobox", "radiogroup", or "listbox", always use "click_option".
- Return ONLY valid JSON — no markdown fences, no explanation."""

    # ── Call Claude while keeping the browser session alive ───────────────────
    stop_event = asyncio.Event()
    keepalive_task = asyncio.create_task(_browser_keepalive(page, stop_event))

    try:
        response = await asyncio.to_thread(
            lambda: client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=16000,
                messages=[{"role": "user", "content": prompt}],
            )
        )
    finally:
        stop_event.set()
        try:
            await asyncio.wait_for(keepalive_task, timeout=5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

    raw = response.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        instructions = json.loads(raw)
    except json.JSONDecodeError:
        print("  ⚠  Claude returned non-JSON — skipping field fill")
        print(f"     Preview: {raw[:300]}")
        return

    print(f"  → Claude mapped {len(instructions)} field(s)")

    for inst in instructions:
        label    = inst.get("label", "unknown")
        selector = inst.get("selector", "")
        value    = str(inst.get("value", ""))
        action   = inst.get("action", "fill")
        frame_url = selector_to_frame_url.get(selector, "")

        print(f"    [{action:14s}] {label}: {repr(value)}")

        if DRY_RUN:
            continue

        try:
            await fill_in_frame(page, selector, action, value, label=label, frame_url=frame_url)
        except Exception as e:
            print(f"    ⚠  Could not fill '{label}': {e}")


# ── Per-job workflow ──────────────────────────────────────────────────────────

async def process_job(
    context: BrowserContext,
    url: str,
    applicant_context: str,
    client: anthropic.Anthropic,
    job_num: int,
) -> None:
    page = await context.new_page()
    print(f"\n{'='*60}")
    print(f"  Job {job_num}: {url}")
    print(f"{'='*60}")

    update_job_status(url, 2)  # Mark as "applying"
    print("  → Status set to 2 (applying)")

    try:
        await page.goto(url, wait_until="networkidle", timeout=30_000)
        await page.wait_for_timeout(1_500)

        # ── Scrape job info before navigating away ────────────────────────────
        job_context = await scrape_job_info(page)
        print(f"  → Job info captured: {job_context.splitlines()[0]}")

        # ── Click Apply ───────────────────────────────────────────────────────
        apply_btn = page.locator(
            "a:has-text('Apply Now'), a:has-text('Apply'), "
            "button:has-text('Apply Now'), button:has-text('Apply')"
        ).first

        if await apply_btn.count() == 0:
            print("  ⚠  No Apply button found — skipping")
            return

        print("  → Clicking Apply Now...")
        await apply_btn.click()
        await page.wait_for_load_state("networkidle", timeout=30_000)
        await page.wait_for_timeout(2_000)
        print(f"  → Application page: {page.url}")

        # ── Handle SSO / login wall before proceeding ─────────────────────────
        await _handle_login_wall(page)
        print(f"  → Post-login page: {page.url}")

        # ── Expand all sections ───────────────────────────────────────────────
        await click_expand_all_link(page)
        # Give SuccessFactors time to finish rendering all expanded sections
        # before we snapshot the field list.
        await page.wait_for_timeout(3_000)

        # ── Fill all form fields ──────────────────────────────────────────────
        await analyze_and_fill(page, client, applicant_context, job_context)

        # ── Click Apply (search parent frame first, then all frames) ─────────
        APPLY_SELECTOR = (
            "button:has-text('Apply'), "
            "a:has-text('Apply'), "
            "input[type='submit'], "
            "button[type='submit']"
        )
        # The Apply button lives on the parent iframe — check page.main_frame
        # first, then walk child frames so we never miss it.
        apply_final = None
        apply_frame = None
        for frame in [page.main_frame] + page.frames[1:]:
            try:
                # Log all buttons/submits in this frame to aid debugging
                candidates = await frame.query_selector_all(
                    "button, input[type='submit'], a[role='button']"
                )
                for c in candidates:
                    try:
                        txt = (await c.inner_text()).strip()
                        vis = await c.is_visible()
                        if txt or vis:
                            print(f"    [frame {frame.url[:60]}] candidate: {repr(txt)} visible={vis}")
                    except Exception:
                        pass

                loc = frame.locator(APPLY_SELECTOR).first
                if await loc.count() > 0 and await loc.is_visible():
                    apply_final = loc
                    apply_frame = frame
                    break
            except Exception:
                pass

        if apply_final is None:
            print("  ⚠  No Apply button found in any frame — done")
        else:
            btn_text = (await apply_final.inner_text()).strip()
            frame_info = f" (frame: {apply_frame.url})" if apply_frame else ""
            if DRY_RUN:
                print(f"  [DRY RUN] Would click '{btn_text}'{frame_info} — not submitting.")
            else:
                confirm = input(
                    f"\n  ⚡ Ready to click '{btn_text}' for job {job_num}. Proceed? (yes/no): "
                ).strip().lower()
                if confirm != "yes":
                    print("  ✗ Skipped by user.")
                else:
                    await apply_final.click()
                    await page.wait_for_load_state("networkidle", timeout=30_000)
                    update_job_status(url, 1)  # Mark as "applied"
                    print(f"  ✓ Application submitted for job {job_num}! Status set to 1 (applied)")

    except Exception as e:
        print(f"  ✗ Error on job {job_num}: {e}")
    finally:
        await page.close()


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    profile = load_profile()
    all_jobs = load_jobs()
    pending_jobs = [j for j in all_jobs if j["status"] == 0]
    applicant_context = build_applicant_context(profile)
    resume_text = load_resume_pdf()
    if resume_text:
        applicant_context += f"\n\nResume (from LinkedIn PDF):\n{resume_text}"
    client = anthropic.Anthropic()

    name = profile["sharing_contexts"]["public"]["data"]["name"]

    print("\n" + "=" * 60)
    print("  Supermicro Job Application Agent")
    print(f"  Applicant : {name}")
    print(f"  Jobs total: {len(all_jobs)}  (pending: {len(pending_jobs)}, applied: {sum(1 for j in all_jobs if j['status'] == 1)}, in-progress: {sum(1 for j in all_jobs if j['status'] == 2)})")
    print(f"  Mode      : {'DRY RUN — preview only, nothing submitted' if DRY_RUN else '⚡ LIVE — will submit after confirmation'}")
    print("=" * 60)

    if not pending_jobs:
        print("\n  All jobs already applied. Nothing to do.")
        return

    if not DRY_RUN:
        print("\n⚠  LIVE mode selected. Applications will be submitted.")
        guard = input("  Type 'confirm' to proceed: ").strip()
        if guard != "confirm":
            print("Aborted.")
            return

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, args=["--start-maximized"])
        context = await browser.new_context(viewport=None)

        # Give the user a chance to log in before automation starts
        setup_page = await context.new_page()
        await setup_page.goto("https://jobs.supermicro.com", wait_until="networkidle")
        print("\n  Browser is open. Log in to jobs.supermicro.com if needed.")
        input("  Press Enter when ready to start applying...")
        await setup_page.close()

        for i, job in enumerate(pending_jobs, 1):
            await process_job(context, job["url"], applicant_context, client, i)
            if i < len(pending_jobs):
                await asyncio.sleep(2)

        print(f"\n{'='*60}")
        print("  All jobs processed.")
        input("  Press Enter to close the browser...")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
