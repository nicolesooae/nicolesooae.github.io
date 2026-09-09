/* Shared behaviour for every page: theme toggle, missing-image placeholders,
   image zoom, copy buttons on code, TOC highlighting, and print prep. */
(function () {
    const root = document.documentElement;

    // ---- Theme toggle (remembers your choice) ----
    try {
        const saved = localStorage.getItem("theme");
        if (saved) root.setAttribute("data-theme", saved);
    } catch (e) { }
    function isDark() {
        const t = root.getAttribute("data-theme");
        if (t) return t === "dark";
        return window.matchMedia("(prefers-color-scheme: dark)").matches;
    }
    function paintToggle() {
        document.querySelectorAll(".theme-toggle").forEach(b => {
            b.textContent = isDark() ? "☀" : "☾";
            b.setAttribute("aria-label", isDark() ? "Switch to light mode" : "Switch to dark mode");
        });
    }
    document.addEventListener("DOMContentLoaded", () => {
        paintToggle();
        document.querySelectorAll(".theme-toggle").forEach(b =>
            b.addEventListener("click", () => {
                const next = isDark() ? "light" : "dark";
                root.setAttribute("data-theme", next);
                try { localStorage.setItem("theme", next); } catch (e) { }
                paintToggle();
            })
        );

        // ---- Missing images: show a labelled placeholder instead of a broken icon ----
        // Drop a file with the same name into media/ and it appears automatically.
        document.querySelectorAll("figure img").forEach(img => {
            const swap = () => {
                const ph = document.createElement("div");
                ph.className = "img-missing";
                ph.textContent = "Add your plot: " + img.getAttribute("src");
                img.replaceWith(ph);
            };
            if (img.complete && img.naturalWidth === 0) swap();
            else img.addEventListener("error", swap);
        });

        // ---- Click a figure to zoom ----
        const lb = document.createElement("div");
        lb.className = "lightbox";
        lb.innerHTML = "<img alt=''>";
        document.body.appendChild(lb);
        lb.addEventListener("click", () => lb.classList.remove("open"));
        document.addEventListener("keydown", e => { if (e.key === "Escape") lb.classList.remove("open"); });
        document.addEventListener("click", e => {
            const img = e.target.closest("figure img");
            if (!img) return;
            lb.querySelector("img").src = img.src;
            lb.querySelector("img").alt = img.alt;
            lb.classList.add("open");
        });

        // ---- Copy buttons on code blocks ----
        document.querySelectorAll("pre > code").forEach(code => {
            const pre = code.parentElement;
            const wrap = document.createElement("div");
            wrap.className = "code-block";
            pre.replaceWith(wrap);
            wrap.appendChild(pre);
            const btn = document.createElement("button");
            btn.className = "copy-btn";
            btn.textContent = "Copy";
            btn.addEventListener("click", async () => {
                try {
                    await navigator.clipboard.writeText(code.innerText);
                    btn.textContent = "Copied";
                } catch (e) { btn.textContent = "Select + copy"; }
                setTimeout(() => (btn.textContent = "Copy"), 1400);
            });
            wrap.appendChild(btn);
        });

        // ---- Tabs: show one .tab-panel at a time, driven by the URL hash ----
        // A link to any id inside a panel (e.g. #p1-results) opens that panel first.
        const panels = [...document.querySelectorAll(".tab-panel")];
        if (panels.length) {
            const tabs = [...document.querySelectorAll(".tab")];
            const tocLists = [...document.querySelectorAll(".toc ol[data-for]")];
            const show = id => {
                panels.forEach(p => p.classList.toggle("active", p.id === id));
                tabs.forEach(t => {
                    const on = t.getAttribute("href") === "#" + id;
                    t.classList.toggle("active", on);
                    t.setAttribute("aria-selected", on);
                });
                tocLists.forEach(ol => (ol.hidden = ol.dataset.for !== id));
            };
            const route = scroll => {
                const target = location.hash && document.getElementById(decodeURIComponent(location.hash.slice(1)));
                const panel = (target && target.closest(".tab-panel")) || panels[0];
                show(panel.id);
                if (!scroll || !target) return;
                // Switching tabs: jump to the top of the new tab only if we've scrolled past it.
                // A section link (e.g. from the contents list) always jumps to the section.
                if (target === panel && panel.getBoundingClientRect().top > 0) return;
                target.scrollIntoView({ block: "start" });
            };
            route(true);
            window.addEventListener("hashchange", () => route(true));
        }

        // ---- PDF build (build.py opens the page as index.html?pdf): expand everything ----
        if (new URLSearchParams(location.search).has("pdf")) {
            root.classList.add("pdf");
            document.querySelectorAll("details").forEach(d => (d.open = true));
        }

        // ---- Highlight the current section in the table of contents ----
        const links = [...document.querySelectorAll(".toc a[href^='#']")];
        if (links.length && "IntersectionObserver" in window) {
            const byId = new Map(links.map(a => [a.getAttribute("href").slice(1), a]));
            const obs = new IntersectionObserver(entries => {
                entries.forEach(en => {
                    if (en.isIntersecting) {
                        links.forEach(a => a.classList.remove("active"));
                        const a = byId.get(en.target.id);
                        if (a) a.classList.add("active");
                    }
                });
            }, { rootMargin: "-20% 0px -70% 0px" });
            byId.forEach((_, id) => { const el = document.getElementById(id); if (el) obs.observe(el); });
        }
    });

    // ---- Printing to PDF: expand every collapsible so nothing is hidden ----
    window.addEventListener("beforeprint", () => {
        document.querySelectorAll("details").forEach(d => { d.dataset.wasOpen = d.open; d.open = true; });
    });
    window.addEventListener("afterprint", () => {
        document.querySelectorAll("details").forEach(d => { d.open = d.dataset.wasOpen === "true"; });
    });
})();