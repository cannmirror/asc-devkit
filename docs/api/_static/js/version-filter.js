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
  var currentVersion = null;
  var _pageInfo = null;

  function openDropdown(el) {
    el.classList.add('show');
    var menu = el.querySelector('.version-filter__menu');
    if (menu) menu.classList.add('show');
  }

  function closeDropdown(el) {
    el.classList.remove('show');
    var menu = el.querySelector('.version-filter__menu');
    if (menu) menu.classList.remove('show');
  }

  function closeAllDropdowns() {
    document.querySelectorAll('.version-filter__container.show').forEach(function (c) {
      closeDropdown(c);
    });
  }

  function syncDropdownButtons() {
    var containers = document.querySelectorAll('.version-filter__container');
    containers.forEach(function (c) {
      var btn = c.querySelector('.version-filter__menu-item[data-version="' + currentVersion + '"]');
      var name = btn ? btn.getAttribute('data-version-name') : currentVersion;
      var current = c.querySelector('.version-filter__current');
      if (current) current.textContent = name;

      c.querySelectorAll('.version-filter__menu-item').forEach(function (item) {
        if (item.getAttribute('data-version') === currentVersion) {
          item.classList.add('active');
        } else {
          item.classList.remove('active');
        }
      });
    });
  }

  function getSidebar() {
    return document.getElementById('pst-primary-sidebar');
  }

  function getVisibleContainers() {
    var containers = [];
    var sidebar = getSidebar();
    if (sidebar) containers.push(sidebar);
    var contentToc = document.querySelector('.bd-article .toctree-wrapper.compound');
    if (contentToc) containers.push(contentToc);
    return containers;
  }

  function showAllSidebarItems() {
    getVisibleContainers().forEach(function (container) {
      container.querySelectorAll('li[class^="toctree-l"]').forEach(function (li) {
        li.style.display = '';
      });
      container.querySelectorAll('details').forEach(function (d) {
        d.style.display = '';
        d.setAttribute('open', 'open');
      });
    });
  }

  function getPagePathInfo() {
    if (_pageInfo) return _pageInfo;
    var pathname = window.location.pathname.replace(/\\/g, '/');
    var parts = pathname.split('/');
    var htmlIdx = -1;
    for (var i = parts.length - 1; i >= 0; i--) {
      if (parts[i] === 'html' && i > 0 && parts[i - 1] === '_build') {
        htmlIdx = i;
        break;
      }
    }
    var depth = (htmlIdx >= 0) ? (parts.length - htmlIdx - 2) : 0;
    var currentDir = (htmlIdx >= 0) ? parts.slice(htmlIdx + 1, -1).join('/') : '';
    _pageInfo = { depth: Math.max(0, depth), currentDir: currentDir };
    return _pageInfo;
  }

  function getCurrentPageDepth() {
    return getPagePathInfo().depth;
  }

  function getCurrentPageDir() {
    return getPagePathInfo().currentDir;
  }

  function resolveToRootRelative(href) {
    if (!href || href.indexOf('://') !== -1 || href.startsWith('#')) return href;
    var depth = getCurrentPageDepth();
    // Count '../' prefixes
    var up = 0;
    var rest = href;
    while (rest.startsWith('../')) {
      up++;
      rest = rest.substring(3);
    }
    var currentDir = getCurrentPageDir();
    var dirParts = currentDir ? currentDir.split('/') : [];
    // Go up
    var effective = dirParts.slice(0, dirParts.length - up);
    var resolved = effective.concat(rest.split('/'));
    return resolved.filter(function(p) { return p; }).join('/');
  }

  function getLinkHref(li) {
    var link = li.querySelector('a.reference.internal, a.reference.external');
    if (!link) return null;
    var href = link.getAttribute('href');
    if (!href) return null;
    try { return decodeURIComponent(resolveToRootRelative(href.split('#')[0])); } catch (e) { return null; }
  }

  function filterSidebarItems() {
    var hidden = window.VERSION_950_HIDDEN;
    if (!hidden || !hidden.length) return;

    var hiddenSet = {};
    for (var i = 0; i < hidden.length; i++) {
      try { hiddenSet[decodeURIComponent(hidden[i])] = true; } catch (e) { hiddenSet[hidden[i]] = true; }
    }

    getVisibleContainers().forEach(function (container) {
      var allItems = container.querySelectorAll('li[class^="toctree-l"]');

      allItems.forEach(function (li) { li.style.display = ''; });

      var itemsArray = [];
      allItems.forEach(function (li) { itemsArray.push(li); });
      itemsArray.reverse();

      itemsArray.forEach(function (li) {
        var href = getLinkHref(li);
        if (!href || !hiddenSet[href]) return;

        var children = li.querySelectorAll('li[class^="toctree-l"]');
        if (children.length === 0) {
          li.style.display = 'none';
          return;
        }

        var allChildrenHidden = true;
        children.forEach(function (child) {
          if (child.style.display !== 'none') {
            allChildrenHidden = false;
          }
        });

        if (allChildrenHidden) {
          li.style.display = 'none';
        }
      });

      container.querySelectorAll('details').forEach(function (d) {
        var lis = d.querySelectorAll('li[class^="toctree-l"]');
        var hasVisible = false;
        for (var j = 0; j < lis.length; j++) {
          if (lis[j].style.display !== 'none') {
            hasVisible = true;
            break;
          }
        }
        if (hasVisible) {
          d.setAttribute('open', 'open');
          d.style.display = '';
        } else {
          d.removeAttribute('open');
          d.style.display = 'none';
        }
      });
    });
  }

  function syncDownloadButton() {
    var btn = document.getElementById('download-btn');
    if (!btn) return;
    var prefix = '';
    for (var i = 0; i < getCurrentPageDepth(); i++) {
      prefix += '../';
    }
    if (currentVersion === '950') {
      btn.href = prefix + 'download_950.zip';
    } else {
      btn.href = prefix + 'download_all.zip';
    }
  }

  function findProductSupportTables() {
    var tables = [];
    document.querySelectorAll('h2').forEach(function (h2) {
      if (h2.textContent.indexOf('产品支持情况') === -1) return;
      var section = h2.closest('section');
      var table = section ? section.querySelector(':scope > table') : null;
      if (!table) {
        var el = h2.nextElementSibling;
        while (el) {
          if (el.tagName === 'TABLE') { table = el; break; }
          var nested = el.querySelector('table');
          if (nested) { table = nested; break; }
          el = el.nextElementSibling;
        }
      }
      if (table) tables.push(table);
    });
    return tables;
  }

  function is950Row(row) {
    var firstTd = row.querySelector('td:first-child');
    if (!firstTd) return false;
    var text = firstTd.textContent;
    return text.indexOf('Ascend 950PR') !== -1 || text.indexOf('Ascend 950DT') !== -1;
  }

  function filterProductSupportTables() {
    var tables = findProductSupportTables();
    tables.forEach(function (table) {
      var tbody = table.querySelector('tbody');
      if (!tbody) return;
      tbody.querySelectorAll('tr').forEach(function (row) {
        if (is950Row(row)) {
          row.style.display = '';
        } else {
          row.style.display = 'none';
        }
      });
    });
  }

  function showAllProductSupportRows() {
    var tables = findProductSupportTables();
    tables.forEach(function (table) {
      var tbody = table.querySelector('tbody');
      if (!tbody) return;
      tbody.querySelectorAll('tr').forEach(function (row) {
        row.style.display = '';
      });
    });
  }

  function applyFilter(version) {
    currentVersion = version;
    syncDropdownButtons();
    syncDownloadButton();

    try { localStorage.setItem('asc-version', version); } catch (e) {}

    if (version === '全量') {
      showAllSidebarItems();
      showAllProductSupportRows();
    } else {
      filterSidebarItems();
      filterProductSupportTables();
    }
  }

  document.addEventListener('click', function (e) {
    var menuItem = e.target.closest('.version-filter__menu-item');
    if (menuItem) {
      e.stopPropagation();
      var version = menuItem.getAttribute('data-version');
      if (!version) return;
      var container = menuItem.closest('.version-filter__container');
      if (container) closeDropdown(container);
      applyFilter(version);
      return;
    }

    var toggleBtn = e.target.closest('.version-filter__button');
    if (toggleBtn) {
      e.stopPropagation();
      var container = toggleBtn.closest('.version-filter__container');
      if (container.classList.contains('show')) {
        closeDropdown(container);
      } else {
        closeAllDropdowns();
        openDropdown(container);
      }
      return;
    }

    closeAllDropdowns();
  });

  document.addEventListener('DOMContentLoaded', function () {
    var savedVersion;
    try { savedVersion = localStorage.getItem('asc-version'); } catch (e) {}

    if (savedVersion && savedVersion !== '全量') {
      applyFilter(savedVersion);
    } else {
      currentVersion = '全量';
      syncDownloadButton();
    }
  });

  window._versionFilter = {
    reapply: function () {
      if (currentVersion === '全量') {
        showAllSidebarItems();
        showAllProductSupportRows();
      } else if (currentVersion) {
        filterSidebarItems();
        filterProductSupportTables();
      }
    }
  };
})();