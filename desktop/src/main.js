// Entry point: tab wiring + kicks off each view's init/load. Vanilla JS,
// no npm/bundler -- <script type="module"> lets these files import each
// other directly via relative paths.
import { initPostingsView, postingsTabActivated, loadPostings, loadPendingBatches } from "./views/postings.js";
import { initApplicationsView, applicationsTabActivated } from "./views/applications.js";
import { initRoadmapView, loadRoadmap, roadmapTabActivated } from "./views/roadmap.js";
import { initCompaniesView, companiesTabActivated } from "./views/companies.js";
import { initProfileView, profileTabActivated } from "./views/profile.js";

const tabButtons = document.querySelectorAll(".tab-btn");
const tabPanels = {
  postings: document.getElementById("postingsTab"),
  applications: document.getElementById("applicationsTab"),
  roadmap: document.getElementById("roadmapTab"),
  companies: document.getElementById("companiesTab"),
  profile: document.getElementById("profileTab"),
};

tabButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    tabButtons.forEach((b) => b.classList.toggle("active", b === btn));
    const tab = btn.dataset.tab;
    Object.entries(tabPanels).forEach(([name, el]) => (el.hidden = name !== tab));
    if (tab === "postings") postingsTabActivated();
    if (tab === "applications") applicationsTabActivated();
    if (tab === "companies") companiesTabActivated();
    if (tab === "profile") profileTabActivated();
    if (tab === "roadmap") roadmapTabActivated();
  });
});

initPostingsView();
initApplicationsView();
initRoadmapView();
initCompaniesView();
initProfileView();

loadPostings();
loadPendingBatches();
loadRoadmap();
