/* Dropdown: type right away (Peter, 2026-10-17).
   Dash 4 dropdowns open with the focus on the selected option, so typing
   does nothing until you click the search box. This puts the cursor in the
   search box as soon as a dropdown opens. Dash loads every .js file in the
   app's assets folder by itself - no change to the app code needed. */
(function () {
  var SEL = "input.dash-dropdown-search";
  function focusLater(inp) {
    // After Dash has set its own focus on the selected option.
    [0, 30, 120].forEach(function (ms) {
      setTimeout(function () {
        if (inp.isConnected && document.activeElement !== inp) {
          inp.focus({ preventScroll: true });
        }
      }, ms);
    });
  }
  new MutationObserver(function (muts) {
    for (var i = 0; i < muts.length; i++) {
      var added = muts[i].addedNodes;
      for (var j = 0; j < added.length; j++) {
        var n = added[j];
        if (n.nodeType !== 1) continue;
        var inp = n.matches(SEL) ? n : n.querySelector(SEL);
        if (inp) { focusLater(inp); return; }
      }
    }
  }).observe(document.documentElement, { childList: true, subtree: true });
})();
