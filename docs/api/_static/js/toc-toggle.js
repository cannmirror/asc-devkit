// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

(function () {
  'use strict';

  var TOC_DATA = null;
  var TOC_LOADING = false;
  var TOC_READY = null;
  var TOC_PENDING_RESOLVERS = [];
  var CURRENT_SECTION = '';
  var CURRENT_PAGE = '';
  var USE_AJAX = window.location.protocol !== 'file:';
  var contentRoot = document.documentElement.getAttribute('data-content_root') || '.';
  var BUILD_ROOT = null;

  function loadToc() {
    if (TOC_READY) return TOC_READY;
    TOC_READY = new Promise(function (resolve) {
      if (TOC_DATA) { resolve(TOC_DATA); return; }
      if (typeof TOC_DATA_EMBEDDED !== 'undefined') {
        TOC_DATA = TOC_DATA_EMBEDDED;
        resolve(TOC_DATA);
        return;
      }
      if (TOC_LOADING) {
        TOC_PENDING_RESOLVERS.push(resolve);
        return;
      }
      TOC_LOADING = true;
      TOC_PENDING_RESOLVERS.push(resolve);
      fetch(contentRoot + '_static/toc_data.json', { cache: 'force-cache' })
        .then(function (r) { return r.json(); })
        .then(function (d) {
          TOC_DATA = d;
          TOC_PENDING_RESOLVERS.forEach(function (r) { r(d); });
          TOC_PENDING_RESOLVERS = [];
        })
        .catch(function () {
          USE_AJAX = false;
          TOC_PENDING_RESOLVERS.forEach(function (r) { r(null); });
          TOC_PENDING_RESOLVERS = [];
        });
    });
    return TOC_READY;
  }

  function findSection(file, entries) {
    entries = entries || TOC_DATA;
    if (!entries) return null;
    for (var i = 0; i < entries.length; i++) {
      if (entries[i].f === file) return entries[i];
      if (entries[i].e && entries[i].e.length > 0) {
        var found = findSection(file, entries[i].e);
        if (found) return found;
      }
    }
    return null;
  }

  function getBuildRoot() {
    if (BUILD_ROOT !== null) return BUILD_ROOT;
    var depth = (contentRoot.match(/\.\.\//g) || []).length;
    var parts = decodeURIComponent(window.location.pathname)
      .replace(/(\/index)?\.html$/, '').replace(/\/+$/, '').split('/');
    BUILD_ROOT = parts.slice(0, Math.max(0, parts.length - depth)).join('/');
    if (BUILD_ROOT) BUILD_ROOT += '/';
    return BUILD_ROOT;
  }

  function tocFileFromHref(href) {
    var a = document.createElement('a');
    a.href = href;
    var path = decodeURIComponent(a.pathname);
    if (path.indexOf(getBuildRoot()) === 0)
      path = path.substring(getBuildRoot().length);
    return path.replace(/(\/index)?\.html$/, '').replace(/\/$/, '');
  }

  function pageUrl(file) {
    return contentRoot + file + '.html';
  }

  function buildTocUl(entries, depth, maxDepth) {
    depth = depth || 1;
    var ul = document.createElement('ul');
    ul.className = 'nav bd-sidenav';
    ul.setAttribute('role', 'tree');
    if (depth === 1) ul.classList.add('current');

    (entries || []).forEach(function (entry) {
      var li = document.createElement('li');
      li.className = 'toctree-l' + depth;
      li.setAttribute('role', 'treeitem');

      var isCurrentPage = CURRENT_PAGE && entry.f === CURRENT_PAGE;
      if (isCurrentPage) { li.classList.add('current', 'active'); li.setAttribute('aria-current', 'page'); }

      var a = document.createElement('a');
      a.className = 'reference internal';
      if (isCurrentPage) a.classList.add('current');
      a.href = pageUrl(entry.f);
      a.textContent = (entry.t || entry.f).replace(/\\(.)/g, '$1');

      if (entry.e && entry.e.length > 0 && (!maxDepth || depth < maxDepth)) {
        li.classList.add('has-children');
        li.setAttribute('aria-expanded', String(isCurrentPage));
        var details = document.createElement('details');
        if (isCurrentPage) details.setAttribute('open', 'open');
        var summary = document.createElement('summary');
        summary.setAttribute('role', 'button');
        summary.setAttribute('aria-expanded', String(isCurrentPage));
        var toggle = document.createElement('span');
        toggle.className = 'toctree-toggle';
        toggle.setAttribute('role', 'presentation');
        toggle.innerHTML = '<i class="fa-solid fa-chevron-down"></i>';
        summary.appendChild(toggle);
        details.appendChild(summary);
        details.appendChild(buildTocUl(entry.e, depth + 1, maxDepth));
        li.appendChild(a);
        li.appendChild(details);
      } else {
        li.appendChild(a);
      }
      ul.appendChild(li);
    });
    return ul;
  }

  function rebuildSidebar(sectionFile) {
    var section = findSection(sectionFile);
    if (!section) return;
    CURRENT_SECTION = sectionFile;

    var tocContainer = document.querySelector('#pst-primary-sidebar nav.bd-links');
    if (!tocContainer) return;
    while (tocContainer.firstChild) tocContainer.removeChild(tocContainer.firstChild);

    var title = document.createElement('p');
    title.className = 'bd-links__title';
    title.setAttribute('role', 'heading');
    title.setAttribute('aria-level', '1');
    title.textContent = 'Section Navigation';
    tocContainer.appendChild(title);

    var tocItem = document.createElement('div');
    tocItem.className = 'bd-toc-item navbar-nav';
    tocItem.appendChild(buildTocUl(section.e || [], 1));
    tocContainer.appendChild(tocItem);
    tocContainer.classList.remove('bd-links--collapsed');

    try { (window._versionFilter || {}).reapply(); } catch (e) {}
  }

  function loadPage(url) {
    if (!USE_AJAX) return Promise.reject(new Error('no ajax'));
    return fetch(url, { cache: 'force-cache' })
      .then(function (r) { return r.text(); })
      .then(function (html) {
        var doc = new DOMParser().parseFromString(html, 'text/html');
        var newTitle = doc.title;
        if (newTitle) document.title = newTitle;
        var newArticle = doc.querySelector('.bd-article-container');
        if (!newArticle) throw new Error('no article');
        var oldArticle = document.querySelector('.bd-article-container');
        if (!oldArticle) throw new Error('no article target');
        oldArticle.innerHTML = newArticle.innerHTML;
        populateContentToc();
        window.scrollTo(0, 0);
      });
  }

  function highlightHeader(sectionFile) {
    var sidebar = document.getElementById('pst-primary-sidebar');
    if (!sidebar) return;
    sidebar.querySelectorAll('.sidebar-header-items__center li.nav-item').forEach(function (item) {
      var link = item.querySelector('a.nav-link.nav-internal');
      if (!link) return;
      var file = tocFileFromHref(link.getAttribute('href'));
      if (file === sectionFile) {
        item.classList.add('current', 'active');
      } else {
        item.classList.remove('current', 'active');
      }
    });
  }

  function navigateTo(sectionFile, pageFile) {
    CURRENT_SECTION = sectionFile;
    CURRENT_PAGE = pageFile || sectionFile;
    var url = pageUrl(CURRENT_PAGE);
    return loadToc().then(function () {
      if (!TOC_DATA) throw new Error('no toc');
      return loadPage(url).then(function () {
        rebuildSidebar(sectionFile);
        highlightHeader(sectionFile);
        try { history.pushState({ s: sectionFile, p: CURRENT_PAGE }, '', url); } catch (e) {}
      });
    });
  }

  function handleSectionClick(file, e) {
    if (e) { e.preventDefault(); e.stopPropagation(); }

    if (file === CURRENT_SECTION) {
      var tc = document.querySelector('#pst-primary-sidebar nav.bd-links');
      if (tc) {
        tc.classList.toggle('bd-links--collapsed');
      }
      return;
    }

    navigateTo(file, file).catch(function () {
      window.location.href = pageUrl(file);
    });
  }

  function handleTocClick(e) {
    var link = e.target.closest('a.reference.internal, a.reference.external');
    if (!link) return;
    var li = link.closest('li');
    if (!li || !li.classList.contains('has-children')) return;
    var details = li.querySelector('details');
    if (!details) return;
    e.preventDefault();
    e.stopPropagation();
    e.stopImmediatePropagation();
    if (details.hasAttribute('open')) {
      details.removeAttribute('open');
      li.setAttribute('aria-expanded', 'false');
    } else {
      details.setAttribute('open', 'open');
      li.setAttribute('aria-expanded', 'true');
    }
  }

  function detectCurrentSection() {
    try {
      var rel;
      if (typeof DOCUMENTATION_OPTIONS !== 'undefined' && DOCUMENTATION_OPTIONS.pagename) {
        rel = DOCUMENTATION_OPTIONS.pagename;
      } else {
        var path = decodeURIComponent(window.location.pathname);
        if (path.indexOf(getBuildRoot()) === 0)
          path = path.substring(getBuildRoot().length);
        rel = path.replace(/(\/index)?\.html$/, '').replace(/\/+$/, '');
      }
      if (rel.indexOf('/') > 0) {
        var dir = rel.split('/')[0];
        CURRENT_SECTION = dir + '/' + dir;
      }
      CURRENT_PAGE = rel;
    } catch (e) {}
  }

  function populateContentToc() {
    var tocWrapper = document.querySelector('.bd-article .toctree-wrapper.compound');
    if (!tocWrapper) return;
    loadToc().then(function () {
      if (!TOC_DATA) { tocWrapper.style.display = 'none'; return; }
      var pageFile = CURRENT_PAGE;
      if (!pageFile) {
        try {
          if (typeof DOCUMENTATION_OPTIONS !== 'undefined' && DOCUMENTATION_OPTIONS.pagename) {
            pageFile = DOCUMENTATION_OPTIONS.pagename;
          } else {
            var path2 = decodeURIComponent(window.location.pathname);
            if (path2.indexOf(getBuildRoot()) === 0) path2 = path2.substring(getBuildRoot().length);
            pageFile = path2.replace(/(\/index)?\.html$/, '').replace(/\/+$/, '');
          }
        } catch (e) {}
      }
      var node = findSection(pageFile);
      if (!node || !node.e || !node.e.length) { tocWrapper.style.display = 'none'; return; }
      while (tocWrapper.firstChild) tocWrapper.removeChild(tocWrapper.firstChild);
      tocWrapper.appendChild(buildTocUl(node.e, 1, 1));
      tocWrapper.style.display = '';
      try { (window._versionFilter || {}).reapply(); } catch (e) {}
    });
  }

  function handleContentTocClick(e) {
    var link = e.target.closest('a.reference.internal, a.reference.external');
    if (!link) return;
    var li = link.closest('li');
    if (!li || !li.classList.contains('has-children')) return;
    var details = li.querySelector('details');
    if (!details) return;
    e.preventDefault();
    e.stopPropagation();
    e.stopImmediatePropagation();
    if (details.hasAttribute('open')) {
      details.removeAttribute('open');
      li.setAttribute('aria-expanded', 'false');
    } else {
      details.setAttribute('open', 'open');
      li.setAttribute('aria-expanded', 'true');
    }
  }

  function init() {
    detectCurrentSection();

    document.addEventListener('DOMContentLoaded', function () {
      var sidebar = document.getElementById('pst-primary-sidebar');
      if (!sidebar) return;

      sidebar.querySelectorAll('.sidebar-header-items__center a.nav-link.nav-internal').forEach(function (link) {
        link.addEventListener('click', function (e) {
          var file = tocFileFromHref(link.getAttribute('href'));
          if (file) handleSectionClick(file, e);
        });
      });

      sidebar.addEventListener('click', handleTocClick);

      var article = document.querySelector('.bd-article');
      if (article) article.addEventListener('click', handleContentTocClick);

      populateContentToc();
    });

    window.addEventListener('popstate', function (e) {
      if (!USE_AJAX) { window.location.reload(); return; }
      if (e.state && e.state.s) {
        CURRENT_SECTION = e.state.s;
        CURRENT_PAGE = e.state.p;
        loadToc().then(function () {
        rebuildSidebar(e.state.s);
        highlightHeader(e.state.s);
        loadPage(pageUrl(e.state.p)).then(function () {
          populateContentToc();
          try { (window._versionFilter || {}).reapply(); } catch (e) {}
        }).catch(noop);
      });
      } else {
        window.location.reload();
      }
    });
  }

  function noop() {}

  init();
})();
