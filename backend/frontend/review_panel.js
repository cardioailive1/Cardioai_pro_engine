// Shows the prediction reviewer's `review` block (backend/review/reviewer.py) under a result card.
// Plain text only (textContent): nothing from the review is ever injected as HTML. Read-only: it explains the score, it does not change it.
(function (root) {
  function el(doc, tag, cls, text) {
    const e = doc.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = String(text);
    return e;
  }
  function list(doc, items, cls) {
    const ul = el(doc, "ul", cls);
    items.forEach(t => ul.appendChild(el(doc, "li", null, t)));
    return ul;
  }
  function fold(doc, title, items) {
    const d = el(doc, "details", "rv-fold");
    d.appendChild(el(doc, "summary", null, title));
    d.appendChild(list(doc, items));
    return d;
  }
  // box: the container element; review: prediction.review (or undefined to hide the panel)
  function renderReview(box, review) {
    if (!box) return;
    const doc = box.ownerDocument;
    box.textContent = "";
    if (!review) { box.hidden = true; return; }
    box.hidden = false;
    const status = ["ok", "check", "warning"].includes(review.status) ? review.status : "check";
    const head = el(doc, "div", "rv-head");
    head.appendChild(el(doc, "strong", null, "Review of this score"));
    head.appendChild(el(doc, "span", `rv-badge ${status}`, status === "ok" ? "No input problems" : status === "warning" ? "Needs attention" : "Check"));
    box.appendChild(head);
    if (review.what_the_tier_means) box.appendChild(el(doc, "p", "rv-tier", review.what_the_tier_means));
    const checks = Array.isArray(review.checks) ? review.checks : [];
    if (checks.length) {
      const ul = el(doc, "ul", "rv-checks");
      checks.forEach(c => ul.appendChild(el(doc, "li", `rv-${["info", "check", "warning"].includes(c.level) ? c.level : "check"}`, c.text)));
      box.appendChild(ul);
    }
    if (Array.isArray(review.known_limits) && review.known_limits.length) box.appendChild(fold(doc, "Known limits of this model", review.known_limits));
    if (Array.isArray(review.for_the_reader) && review.for_the_reader.length) box.appendChild(fold(doc, "For the reader", review.for_the_reader));
    box.appendChild(el(doc, "div", "rv-note", review.disclaimer || "This review explains the score; it is not a diagnosis and does not decide care."));
    if (review.wording_status && !String(review.wording_status).startsWith("approved")) box.appendChild(el(doc, "div", "rv-draft", review.wording_status));
  }
  root.renderReview = renderReview;
  if (typeof module !== "undefined" && module.exports) module.exports = { renderReview };
})(typeof window !== "undefined" ? window : globalThis);
