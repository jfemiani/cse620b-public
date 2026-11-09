const tabs = [...document.querySelectorAll('[role="tab"]')];
const panels = [...document.querySelectorAll('[role="tabpanel"]')];
const viewer = document.querySelector("#viewer");
const frame = document.querySelector("#material-frame");
const heading = document.querySelector("#viewer-title");
const external = document.querySelector("#open-material");
const error = document.querySelector("#viewer-error");
let selectedMaterial;

function showMaterial(link, { scroll = false } = {}) {
  selectedMaterial?.removeAttribute("aria-current");
  selectedMaterial = link;
  link.setAttribute("aria-current", "true");
  const title = `${link.closest(".resource").querySelector("h3").textContent}: ${link.textContent}`;
  heading.textContent = title;
  frame.title = title;
  external.href = link.href;
  error.hidden = true;
  if (frame.src !== link.href) frame.src = link.href;
  frame.hidden = false;
  viewer.hidden = false;
  if (scroll) viewer.scrollIntoView({ behavior: "smooth", block: "start" });
}

function activateTab(index, { focus = false, updateHash = true } = {}) {
  tabs.forEach((tab, position) => {
    const active = position === index;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    panels[position].hidden = !active;
  });
  if (focus) tabs[index].focus();
  if (updateHash) history.replaceState(null, "", `#${tabs[index].id}`);
  const firstMaterial = panels[index].querySelector("[data-embed]");
  if (firstMaterial) {
    showMaterial(firstMaterial);
  } else {
    frame.removeAttribute("src");
    frame.hidden = true;
    viewer.hidden = true;
    selectedMaterial?.removeAttribute("aria-current");
    selectedMaterial = undefined;
  }
}

tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => activateTab(index));
  tab.addEventListener("keydown", (event) => {
    let next;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    if (event.key === "ArrowLeft") next = (index + tabs.length - 1) % tabs.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = tabs.length - 1;
    if (next !== undefined) {
      event.preventDefault();
      activateTab(next, { focus: true });
    }
  });
});

document.querySelectorAll("[data-embed]").forEach((link) => {
  link.addEventListener("click", (event) => {
    if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    showMaterial(link, { scroll: true });
  });
});

frame.addEventListener("error", () => {
  error.textContent = "This material could not be loaded. Try the Open separately link.";
  error.hidden = false;
});

document.querySelector("#fullscreen-material").addEventListener("click", async () => {
  try {
    if (document.fullscreenElement) {
      await document.exitFullscreen();
    } else {
      await viewer.requestFullscreen();
    }
  } catch (failure) {
    error.textContent = `Full screen is unavailable: ${failure.message}. Use Open separately instead.`;
    error.hidden = false;
  }
});

function restoreTab() {
  const index = tabs.findIndex((tab) => `#${tab.id}` === location.hash);
  activateTab(index < 0 ? 0 : index, { updateHash: false });
}
window.addEventListener("hashchange", restoreTab);
restoreTab();
