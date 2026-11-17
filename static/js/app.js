/**
 * Rajasthan Royals SupeRR Selector 2.0 — Interactive Client Application
 * Modern, accessible sports-analytics dashboard engine.
 */

// Global High-Pressure Evidence Sample Size Thresholds (Named Constants)
const HP_EVIDENCE_THRESHOLDS = {
  STRONG: 200,   // Neff_HP >= 200
  MODERATE: 100, // 100 <= Neff_HP < 200
  THIN: 100      // Neff_HP < 100
};

// Application Reactive State
const STATE = {
  tab: 'overview',
  audienceMode: 'simple', // 'simple' (Analyst & Fan) or 'tech' (Data Science)
  batView: 'hp_headline', // 'hp_headline' or 'all'
  bowlView: 'hp_headline',
  auctionBucket: 'pending_retention_watchlist',
  intelView: 'batting',
  intelSearch: '',
  filters: {
    search: '',
    role: '',
    availability: '',
    ipl_exp: '',
    pressure_tier: '',
    confidence_tier: '',
    all_rounder_only: false
  },
  compareIds: [1699, 6396, 2933, 7920],
  quickCompare: [],
  sensitivityMode: false, // Supplementary z=1.0 sensitivity view
  sort: { column: null, asc: true }
};

// Plain English Metric Glossary
const METRIC_GLOSSARY = {
  pi: {
    simple: "Match Pressure",
    tech: "Pressure Index (PI, 0–100)",
    desc: "Causal pre-delivery difficulty based on required run rate, wickets down, and balls remaining."
  },
  neff_hp: {
    simple: "High-Pressure Opportunities",
    tech: "Effective Sample Size in High Pressure (Neff_HP)",
    desc: "Deliveries faced when PI >= 60, weighted by competition prestige (IPL=1.0, T20I=0.5)."
  },
  shrinkage: {
    simple: "Experience Adjustment",
    tech: "Empirical Bayesian Normal-Normal Shrinkage",
    desc: "Adjusts small-sample streaks toward league baseline to prevent overrating unproven performers."
  },
  pri_lcb: {
    simple: "Conservative Reliability Score",
    tech: "Player Reliability Index — Lower Credible Bound (PRI_lcb)",
    desc: "A conservative 0–100 rating that gives credit for high skill while penalizing low sample size."
  },
  hp_raa: {
    simple: "Runs Above Expected Under Pressure",
    tech: "High-Pressure Runs Above Average (HP-RAA, runs/100b)",
    desc: "Runs scored above contextual expectation in high-pressure chases (lower-bound score z=0.5)."
  },
  hp_impact: {
    simple: "Runs Saved Above Expected Under Pressure",
    tech: "High-Pressure Bowling Impact (HP-Impact, runs/over saved)",
    desc: "Runs prevented compared to average bowler delivering under identical crunch conditions."
  },
  wpa: {
    simple: "Impact on Winning Chances",
    tech: "Win Probability Added (WPA)",
    desc: "Cumulative percentage shift in win probability produced across match-defining moments."
  },
  leverage: {
    simple: "Importance of the Match Situation",
    tech: "Leverage Index (LI)",
    desc: "How much the match win probability can swing on the current delivery. Orthogonal to PI."
  },
  ci: {
    simple: "Ranking Uncertainty Range",
    tech: "95% Match-Cluster Bootstrap Rank Interval",
    desc: "The range of ranks the player is expected to hold across 2,000 simulated tournament reruns."
  }
};

function lbl(key) {
  const g = METRIC_GLOSSARY[key];
  if (!g) return key;
  return STATE.audienceMode === 'simple' ? g.simple : g.tech;
}

function tip(key) {
  const g = METRIC_GLOSSARY[key];
  if (!g) return '';
  return `<span class="kpi-sub" style="cursor:help;text-decoration:underline dotted" title="${g.desc}">ⓘ</span>`;
}

// Global Keyboard Shortcut: Cmd+K / Ctrl+K to focus search
document.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
    e.preventDefault();
    const searchInput = document.getElementById('global-search');
    if (searchInput) {
      searchInput.focus();
      searchInput.select();
    }
  }
  if (e.key === 'Escape') {
    closeModal();
  }
});

// Mobile navigation outside click handler
document.addEventListener('click', (e) => {
  if (document.body.classList.contains('mobile-nav-open') && !e.target.closest('#sidebar') && !e.target.closest('#mobile-menu-btn')) {
    document.body.classList.remove('mobile-nav-open');
    const button = document.getElementById('mobile-menu-btn');
    if (button) button.setAttribute('aria-expanded', 'false');
  }
});

// Audience Mode & Sidebar Toggles
function setAudienceMode(mode) {
  STATE.audienceMode = mode;
  document.getElementById('aud-simple')?.classList.toggle('active', mode === 'simple');
  document.getElementById('aud-tech')?.classList.toggle('active', mode === 'tech');
  renderTab();
}

function toggleSidebar() {
  document.body.classList.toggle('sidebar-collapsed');
}

function toggleMobileNav() {
  const open = document.body.classList.toggle('mobile-nav-open');
  const btn = document.getElementById('mobile-menu-btn');
  if (btn) btn.setAttribute('aria-expanded', open ? 'true' : 'false');
}

function openTab(tabName) {
  STATE.tab = tabName;
  document.querySelectorAll('.nav-link').forEach(el => {
    el.classList.toggle('active', el.dataset.tab === tabName);
  });
  document.body.classList.remove('mobile-nav-open');
  renderTab();
}

// Evidence Tier Badges
function evidenceBadge(badgeText, neffHp) {
  if (neffHp != null) {
    if (neffHp === 0) {
      return `<span class="chip-tier chip-no-hp" title="0 pressure balls faced — No high-pressure evidence">No HP Evidence (0)</span>`;
    } else if (neffHp >= HP_EVIDENCE_THRESHOLDS.STRONG) {
      return `<span class="chip-tier chip-strong" title="Neff_HP >= 200: Substantial high-pressure evidence">Strong Evidence (${Math.round(neffHp)})</span>`;
    } else if (neffHp >= HP_EVIDENCE_THRESHOLDS.MODERATE) {
      return `<span class="chip-tier chip-moderate" title="100 <= Neff_HP < 200: Moderate sample size">Moderate Evidence (${Math.round(neffHp)})</span>`;
    } else {
      return `<span class="chip-tier chip-thin" title="Neff_HP < 100: Thin sample size">Thin Evidence (${Math.round(neffHp)})</span>`;
    }
  }

  const s = String(badgeText || '').toLowerCase();
  if (s.includes('strong')) {
    return `<span class="chip-tier chip-strong">Strong Evidence</span>`;
  } else if (s.includes('moderate')) {
    return `<span class="chip-tier chip-moderate">Moderate Evidence</span>`;
  } else if (s.includes('insufficient') || s.includes('zero') || s.includes('0')) {
    return `<span class="chip-tier chip-no-hp">No HP Evidence</span>`;
  }
  return `<span class="chip-tier chip-thin">${badgeText || 'Thin Evidence'}</span>`;
}

// Availability Status Badges (Strict: Only Confirmed Released = Target)
function statusBadge(status) {
  const s = String(status || '').toUpperCase();
  if (s.includes('CONFIRMED_RELEASED') || s.includes('ACTIONABLE AUCTION TARGET')) {
    return `<span class="badge badge-target">Confirmed Target</span>`;
  } else if (s.includes('PENDING') || s.includes('RELEASED') || s.includes('WATCHLIST') || s.includes('UNVERIFIED')) {
    return `<span class="badge badge-watchlist">Pending Watchlist</span>`;
  } else if (s.includes('RR') || s.includes('RAJASTHAN ROYALS')) {
    return `<span class="badge badge-rr">Rajasthan Royals</span>`;
  } else if (s.includes('RETIRED')) {
    return `<span class="badge badge-retired">Retired</span>`;
  } else if (s.includes('CONTRACTED') || s.includes('OTHER_TEAM') || s.includes('TRADED')) {
    return `<span class="badge badge-contracted">Contracted / Excluded</span>`;
  }
  return `<span class="badge b-info">${status || 'Pending'}</span>`;
}

function teamBadge(team) {
  if (!team || team === 'None' || team === 'Unknown') return '<span class="team-tag">—</span>';
  return `<span class="team-tag">${team}</span>`;
}

// Comparison Tray State & Actions
function toggleQuickCompare(cid, name) {
  if (!STATE.quickCompare) STATE.quickCompare = [];
  const idx = STATE.quickCompare.findIndex(x => x.id === cid);
  if (idx >= 0) {
    STATE.quickCompare.splice(idx, 1);
  } else {
    if (STATE.quickCompare.length >= 4) {
      STATE.quickCompare.shift();
    }
    STATE.quickCompare.push({ id: cid, name: name });
  }
  updateQuickCompareDrawer();
}

function updateQuickCompareDrawer() {
  const drawer = document.getElementById('quick-compare-drawer');
  const label = document.getElementById('quick-compare-names');
  if (!drawer || !label) return;
  if (!STATE.quickCompare || STATE.quickCompare.length === 0) {
    drawer.classList.remove('active');
  } else {
    drawer.classList.add('active');
    label.innerHTML = STATE.quickCompare.map(x => `<span class="badge b-strong">${x.name}</span>`).join(' &bull; ');
  }
}

function clearQuickCompare() {
  STATE.quickCompare = [];
  updateQuickCompareDrawer();
}

function executeQuickCompare() {
  if (!STATE.quickCompare || STATE.quickCompare.length === 0) return;
  STATE.compareIds = STATE.quickCompare.map(x => x.id);
  openTab('compare');
}

// Filter Query Builder
function filterQuery(extra = {}) {
  const params = new URLSearchParams();
  const f = { ...STATE.filters, ...extra };
  if (f.discipline) params.set('discipline', f.discipline);
  if (f.search) params.set('search', f.search);
  if (f.role) params.set('role', f.role);
  if (f.availability) params.set('availability', f.availability);
  if (f.ipl_exp) params.set('ipl_exp', f.ipl_exp);
  if (f.pressure_tier) params.set('pressure_tier', f.pressure_tier);
  if (f.confidence_tier) params.set('confidence_tier', f.confidence_tier);
  if (f.all_rounder_only) params.set('all_rounder_only', 'true');
  return params.toString();
}

function filterBarHtml(showAvail = true) {
  return `
    <div class="filter-panel">
      <div class="filter-group">
        <label class="filter-label">Search</label>
        <input class="filter-input" type="search" placeholder="Search player..." value="${STATE.filters.search.replace(/"/g, '&quot;')}" oninput="STATE.filters.search=this.value; renderTab()" />
      </div>
      <div class="filter-group">
        <label class="filter-label">Role</label>
        <select class="filter-select" onchange="STATE.filters.role=this.value; renderTab()">
          <option value="">All Roles</option>
          <option value="Top Order" ${STATE.filters.role==='Top Order'?'selected':''}>Top Order</option>
          <option value="Middle Order" ${STATE.filters.role==='Middle Order'?'selected':''}>Middle Order</option>
          <option value="Finisher" ${STATE.filters.role==='Finisher'?'selected':''}>Finisher</option>
          <option value="Pace Spearhead" ${STATE.filters.role==='Pace Spearhead'?'selected':''}>Pace Spearhead</option>
          <option value="Death Specialist" ${STATE.filters.role==='Death Specialist'?'selected':''}>Death Specialist</option>
          <option value="Spin Container" ${STATE.filters.role==='Spin Container'?'selected':''}>Spin Container</option>
        </select>
      </div>
      ${showAvail ? `
      <div class="filter-group">
        <label class="filter-label">Availability</label>
        <select class="filter-select" onchange="STATE.filters.availability=this.value; renderTab()">
          <option value="">All Availability</option>
          <option value="TARGET" ${STATE.filters.availability==='TARGET'?'selected':''}>Target (Confirmed Released)</option>
          <option value="WATCHLIST" ${STATE.filters.availability==='WATCHLIST'?'selected':''}>Watchlist (Pending / Unverified)</option>
          <option value="CONTRACTED" ${STATE.filters.availability==='CONTRACTED'?'selected':''}>Contracted to Other Franchise</option>
          <option value="RR" ${STATE.filters.availability==='RR'?'selected':''}>Current RR Squad</option>
          <option value="RETIRED" ${STATE.filters.availability==='RETIRED'?'selected':''}>Retired</option>
        </select>
      </div>` : ''}
      <div class="filter-group">
        <label class="filter-label">Pressure Exp.</label>
        <select class="filter-select" onchange="STATE.filters.pressure_tier=this.value; renderTab()">
          <option value="">All Levels</option>
          <option value="robust" ${STATE.filters.pressure_tier==='robust'?'selected':''}>Robust (Neff_HP &ge; 30)</option>
          <option value="limited" ${STATE.filters.pressure_tier==='limited'?'selected':''}>Limited (&lt; 15)</option>
          <option value="zero" ${STATE.filters.pressure_tier==='zero'?'selected':''}>Zero Pressure Balls</option>
        </select>
      </div>
      <div class="filter-group" style="margin-left:auto">
        <button class="pill" onclick="STATE.filters={search:'',role:'',availability:'',ipl_exp:'',pressure_tier:'',confidence_tier:'',all_rounder_only:false}; renderTab()">
          Reset Filters
        </button>
      </div>
    </div>
  `;
}

function handleGlobalSearch(query) {
  STATE.filters.search = query.trim();
  if (['overview', 'batting', 'bowling', 'auction_intelligence'].includes(STATE.tab)) {
    renderTab();
  } else {
    openTab('batting');
  }
}

// Fetch Helper with Error Boundaries
async function fetchJson(url) {
  const resp = await fetch(url);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`API Error ${resp.status} (${url}): ${text.slice(0, 140)}`);
  }
  return await resp.json();
}

// Client-Side Column Sorting Helper
function sortTableData(items, col, isNumber = false) {
  if (STATE.sort.column === col) {
    STATE.sort.asc = !STATE.sort.asc;
  } else {
    STATE.sort.column = col;
    STATE.sort.asc = true;
  }
  return [...items].sort((a, b) => {
    let va = a[col];
    let vb = b[col];
    if (va == null) va = isNumber ? -Infinity : '';
    if (vb == null) vb = isNumber ? -Infinity : '';
    if (isNumber) {
      return STATE.sort.asc ? Number(va) - Number(vb) : Number(vb) - Number(va);
    }
    return STATE.sort.asc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
  });
}

// Master Tab Rendering Engine
async function renderTab() {
  const el = document.getElementById('app-content');
  if (!el) return;

  const pages = {
    overview: ['High-Pressure & Reliability Overview', 'Causal pre-delivery match state evaluations and official competition deliverables for the Rajasthan Royals challenge.', 'Official Deliverables'],
    batting: ['Batter Pressure Scouting', 'Compare demonstrated high-pressure clutch evidence against overall multi-phase consistency.', 'Player Analytics'],
    bowling: ['Bowler Pressure Scouting', 'Assess run prevention, yorker execution, and boundary suppression across ordinary and crunch phases.', 'Player Analytics'],
    compare: ['Head-to-Head Player Comparison', 'Side-by-side performance evidence, phase breakdowns, and acquisition pathways for recruitment candidates.', 'Decision Support'],
    pressure_leverage: ['Pressure & Win Impact', 'Demonstrating the mathematical distinction between game difficulty (PI) and win swing leverage (LI).', 'Contextual Modeling'],
    cricket_intelligence: ['Tactical Player Intelligence', 'Observed post-dot ball strike rates, post-boundary bounce-back composure, and 4-question franchise strategies.', 'Advanced Analytics'],
    auction_intelligence: ['Auction & Recruitment Intelligence', 'Strict separation of performance evidence, tactical suitability, and verified BCCI auction availability.', 'Recruitment'],
    rr_squad: ['Rajasthan Royals 2026 Campaign Review', 'Squad performance audit addressing the SMS Jaipur pace economy leak (10.13 RPO) vs. Guwahati spin equilibrium (3/3).', 'Franchise Analysis'],
    trades: ['IPL 2027 Trade Ledger', 'Formally registered franchise player movements tracked separately from unverified rumours and stalled negotiations.', 'Recruitment'],
    model_eval: ['Model Validation & Econometric Sensitivity', 'Out-of-fold calibration, multi-scenario weight sensitivity, bootstrap intervals, and holdout tests.', 'Methodology'],
    methodology: ['Technical Methodology & Limitations', 'Formulas, Kish effective sample sizing, Empirical Bayes shrinkage, zero-leakage, and explicit data boundaries.', 'Methodology']
  };

  const meta = pages[STATE.tab] || ['Rajasthan Royals SupeRR Selector 2.0', '', 'Scouting'];
  const renderers = {
    overview: renderOverview,
    batting: renderBattingScouting,
    bowling: renderBowlingScouting,
    compare: renderCompare,
    pressure_leverage: renderPressureLeverage,
    cricket_intelligence: renderCricketIntelligence,
    auction_intelligence: renderAuctionIntelligence,
    rr_squad: renderRRSquadAudit,
    trades: renderTrades,
    model_eval: renderModelEvaluation,
    methodology: renderMethodology
  };

  el.innerHTML = `
    <div class="card" style="text-align:center; padding: 64px 24px;">
      <div style="font-size:1.1rem; color:var(--text-muted); margin-bottom:12px">Loading ${meta[0]}...</div>
      <div style="width:40px;height:40px;border:3px solid var(--border);border-top-color:var(--rr-pink);border-radius:50%;margin:0 auto;animation:spin 1s linear infinite"></div>
    </div>
  `;

  try {
    const fn = renderers[STATE.tab];
    if (fn) {
      await fn(el);
      el.insertAdjacentHTML('afterbegin', `
        <section class="view-heading">
          <div class="view-eyebrow">${meta[2]}</div>
          <h1>${meta[0]}</h1>
          <p>${meta[1]}</p>
        </section>
      `);
    }
  } catch (err) {
    console.error('Error rendering tab:', err);
    el.innerHTML = `
      <div class="alert alert-danger" style="margin:24px 0;">
        <div>
          <strong>Error loading analytical view:</strong> ${err.message}
          <div style="margin-top:10px">
            <button class="pill active" onclick="renderTab()">Retry Loading</button>
          </div>
        </div>
      </div>
    `;
  }
}

// ==========================================================================
// 1. OVERVIEW / MASTER EXECUTIVE DASHBOARD
// ==========================================================================
async function renderOverview(el) {
  const [d, dims, wp, intel, off] = await Promise.all([
    fetchJson('/api/overview'),
    fetchJson('/api/evaluation-dimensions'),
    fetchJson('/api/win-probability'),
    fetchJson('/api/rr-tactical-fit'),
    fetchJson('/api/rankings/official-top10')
  ]);

  const rawBat = off.deliverable_01?.batters || dims.batting?.top10_high_pressure_robust || [];
  const rawBowl = off.deliverable_01?.bowlers || dims.bowling?.top10_high_pressure_robust || [];

  // Supplementary sensitivity (z = 1.0) re-ranking for the SAME 10 players
  let topBat = rawBat;
  let topBowl = rawBowl;

  if (STATE.sensitivityMode) {
    topBat = [...rawBat].map(r => {
      const lcb = r.hp_raa_lcb ?? r.hp_score_lcb_runs_per_100b;
      const post = r.hp_raa_post ?? r.hp_score_post_runs_per_100b;
      const score_z10 = (lcb != null && post != null) ? (2 * lcb - post) : lcb;
      return { ...r, score_z10 };
    }).sort((a, b) => b.score_z10 - a.score_z10);

    topBowl = [...rawBowl].map(r => {
      const lcb = r.hp_impact_lcb ?? r.hp_impact_lcb_runs_saved_per_ov;
      const post = r.hp_impact_post ?? r.hp_impact_post_runs_saved_per_ov;
      const score_z10 = (lcb != null && post != null) ? (2 * lcb - post) : lcb;
      return { ...r, score_z10 };
    }).sort((a, b) => b.score_z10 - a.score_z10);
  }

  el.innerHTML = `
    <!-- Top Executive Metric Strip -->
    <div class="kpi-strip">
      <div class="kpi-cell" style="--accent-color:var(--cyan)">
        <div class="kpi-label">Audited Deliveries ${tip('pi')}</div>
        <div class="kpi-val" style="color:var(--cyan)">${d.universe_counts.primary_deliveries.toLocaleString()}</div>
        <div class="kpi-sub">867 Matches (IPL 1.0 + T20I 0.5)</div>
      </div>
      <div class="kpi-cell" style="--accent-color:var(--purple)">
        <div class="kpi-label">Evaluated Universe ${tip('shrinkage')}</div>
        <div class="kpi-val">${d.universe_counts.eligible_batters} <span class="kpi-dim">/</span> ${d.universe_counts.eligible_bowlers}</div>
        <div class="kpi-sub">Batters / Bowlers (&ge; 30 balls faced)</div>
      </div>
      <div class="kpi-cell" style="--accent-color:var(--rr-pink)">
        <div class="kpi-label">RR 2026 Campaign</div>
        <div class="kpi-val" style="color:var(--rr-pink);font-size:1.15rem">${intel.rr_2026_record.split(',')[0]}</div>
        <div class="kpi-sub">Guwahati 3–0 vs Jaipur 1–4</div>
      </div>
      <div class="kpi-cell" style="--accent-color:var(--gold)">
        <div class="kpi-label">Pressure vs Leverage ${tip('leverage')}</div>
        <div class="kpi-val" style="color:var(--gold)">r = +0.01</div>
        <div class="kpi-sub">Orthogonal Signals (Inn 1: -0.37, Inn 2: +0.08)</div>
      </div>
      <div class="kpi-cell" style="--accent-color:var(--green)">
        <div class="kpi-label">Auction Target Status</div>
        <div class="kpi-val" style="color:var(--green)">0 Confirmed</div>
        <div class="kpi-sub">Nov 17 Retention / Dec 16 Auction</div>
      </div>
    </div>

    <!-- Statistical Disclosure Notice on High-Pressure Rankings -->
    <div class="alert alert-info" style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px">
      <div>
        <strong style="color:var(--cyan)">Statistical Fairness Disclosure:</strong>
        High-pressure scores represent the <strong>conservative lower-bound score (z = 0.5)</strong> under Empirical Bayesian shrinkage, <strong>not a 95% confidence interval, percentage or probability</strong>.
        <em>Adjacent ranks are not statistically separable; read close ranks as a group.</em>
      </div>
      <div>
        <button class="btn-sensitivity-toggle ${STATE.sensitivityMode ? 'active' : ''}" onclick="STATE.sensitivityMode = !STATE.sensitivityMode; renderTab();">
          <svg class="icon" viewBox="0 0 24 24" style="width:14px;height:14px"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
          ${STATE.sensitivityMode ? 'Revert to Official Baseline (z = 0.5)' : 'Supplementary Sensitivity (z = 1.0)'}
        </button>
      </div>
    </div>

    ${STATE.sensitivityMode ? `
      <div class="alert alert-warn">
        <strong>Supplementary Sensitivity Active (z = 1.0):</strong> Showing re-ordered ranking of the <strong>SAME 10 official players</strong> under a stricter penalty bound.
        <em>This is a diagnostic robustness check only and does not replace or alter the official competition baseline.</em>
      </div>
    ` : ''}

    <!-- PRIMARY DELIVERABLE: TOP 10 HIGH PRESSURE SHOWCASE -->
    <div class="card-title">
      <span class="icon-title">
        <svg class="icon" style="color:var(--rr-pink);width:20px;height:20px" viewBox="0 0 24 24"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
        <span>PRIMARY DELIVERABLE: TOP 10 PLAYERS UNDER HIGH PRESSURE (PI &ge; 60, Neff_HP &ge; 30)</span>
      </span>
      <span class="kpi-sub">Sample size tiers: Strong (&ge;200), Moderate (100–200), Thin (&lt;100)</span>
    </div>

    <div class="grid-2">
      <!-- TOP 10 BATTERS -->
      <div class="card" style="border-top:3px solid var(--rr-pink)">
        <div class="card-title" style="color:var(--rr-pink);margin-bottom:6px">
          <span>Top 10 Most Reliable Batters Under Pressure</span>
          <span style="font-size:0.75rem;color:var(--text-dim);font-weight:400">Runs / 100 Balls Above Expected</span>
        </div>
        <p class="kpi-sub" style="margin-bottom:14px">
          Official baseline lower-bound score ($z = 0.5$). Measures run production in steep run chases and high-pressure collapses.
        </p>
        <div>
          ${topBat.map((r, i) => {
            const lcb = r.hp_raa_lcb ?? r.hp_score_lcb_runs_per_100b;
            const post = r.hp_raa_post ?? r.hp_score_post_runs_per_100b;
            const neff = r.neff_hp ?? r.neff_hp_opportunities;
            const scoreToDisplay = STATE.sensitivityMode ? r.score_z10 : lcb;
            return `
              <div class="scannable-row" onclick="openPlayer(${r.canonical_player_id})">
                <div class="scannable-row-main">
                  <div class="scannable-row-left">
                    <div class="scannable-rank ${i < 3 ? 'rank-top3' : ''}">#${i+1}</div>
                    <div>
                      <div class="scannable-name-row">
                        <span class="scannable-name">${r.player_name}</span>
                        ${evidenceBadge(r.evidence_badge, neff)}
                        <button class="btn-quick-cmp" onclick="event.stopPropagation(); toggleQuickCompare(${r.canonical_player_id}, '${r.player_name.replace(/'/g, "\\'")}')" title="Add to comparison">
                          Compare
                        </button>
                      </div>
                      <div class="scannable-score-sub">
                        LCB (z=0.5): <strong>+${Number(lcb).toFixed(2)}</strong> &bull; Mean: +${Number(post).toFixed(2)} &bull; ${Math.round(neff)} balls
                        ${STATE.sensitivityMode ? `&bull; <span style="color:var(--gold)">z=1.0: +${Number(r.score_z10).toFixed(2)}</span>` : ''}
                      </div>
                    </div>
                  </div>
                  <div class="scannable-row-right">
                    <div class="scannable-metric-num">+${Number(scoreToDisplay).toFixed(2)}</div>
                    <div class="scannable-metric-sub">runs / 100b</div>
                  </div>
                </div>
                <div class="scannable-analyst-snippet">
                  <strong>Analyst Verdict:</strong> ${r.analyst_blurb}
                </div>
              </div>
            `;
          }).join('')}
        </div>
      </div>

      <!-- TOP 10 BOWLERS -->
      <div class="card" style="border-top:3px solid var(--cyan)">
        <div class="card-title" style="color:var(--cyan);margin-bottom:6px">
          <span>Top 10 Most Reliable Bowlers Under Pressure</span>
          <span style="font-size:0.75rem;color:var(--text-dim);font-weight:400">Runs Saved / Over Above Expected</span>
        </div>
        <p class="kpi-sub" style="margin-bottom:14px">
          Official baseline lower-bound score ($z = 0.5$). Measures run prevention and dot induction during crunch death overs.
        </p>
        <div>
          ${topBowl.map((r, i) => {
            const lcb = r.hp_impact_lcb ?? r.hp_impact_lcb_runs_saved_per_ov;
            const post = r.hp_impact_post ?? r.hp_impact_post_runs_saved_per_ov;
            const neff = r.neff_hp ?? r.neff_hp_opportunities;
            const scoreToDisplay = STATE.sensitivityMode ? r.score_z10 : lcb;
            return `
              <div class="scannable-row" onclick="openPlayer(${r.canonical_player_id})">
                <div class="scannable-row-main">
                  <div class="scannable-row-left">
                    <div class="scannable-rank ${i < 3 ? 'rank-top3' : ''}" style="${i < 3 ? 'background:linear-gradient(135deg, var(--cyan) 0%, #254aa5 100%);border-color:rgba(56,189,248,0.4);box-shadow:0 0 10px var(--cyan-glow)' : ''}">#${i+1}</div>
                    <div>
                      <div class="scannable-name-row">
                        <span class="scannable-name">${r.player_name}</span>
                        ${evidenceBadge(r.evidence_badge, neff)}
                        <button class="btn-quick-cmp" onclick="event.stopPropagation(); toggleQuickCompare(${r.canonical_player_id}, '${r.player_name.replace(/'/g, "\\'")}')" title="Add to comparison">
                          Compare
                        </button>
                      </div>
                      <div class="scannable-score-sub">
                        LCB (z=0.5): <strong>+${Number(lcb).toFixed(3)}</strong> &bull; Mean: +${Number(post).toFixed(3)} &bull; ${Math.round(neff)} balls
                        ${STATE.sensitivityMode ? `&bull; <span style="color:var(--gold)">z=1.0: +${Number(r.score_z10).toFixed(3)}</span>` : ''}
                      </div>
                    </div>
                  </div>
                  <div class="scannable-row-right">
                    <div class="scannable-metric-num" style="color:var(--cyan)">+${Number(scoreToDisplay).toFixed(3)}</div>
                    <div class="scannable-metric-sub">runs saved / ov</div>
                  </div>
                </div>
                <div class="scannable-analyst-snippet" style="border-left-color:var(--cyan)">
                  <strong>Analyst Verdict:</strong> ${r.analyst_blurb}
                </div>
              </div>
            `;
          }).join('')}
        </div>
      </div>
    </div>

    <!-- Powerplay Exposure Paradox Alert -->
    <div class="card" style="margin-top:24px;border-left:4px solid var(--gold);background:rgba(245, 158, 11, 0.04)">
      <div class="card-title" style="color:var(--gold);margin-bottom:8px">
        <span class="icon-title">
          <svg class="icon" viewBox="0 0 24 24"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
          The Powerplay Exposure Paradox: High Overall Rating vs Zero Pressure Experience
        </span>
      </div>
      <p style="font-size:0.88rem;color:var(--text-secondary);margin-bottom:12px">
        Overall reliability (Dimension A) and High-Pressure clutch execution (Dimension B) address completely separate questions. Top-order openers who bat exclusively in overs 1–6 face field restrictions with zero match-pressure index (PI &lt; 20). They must not be miscast as late-innings finishers.
      </p>
      <div style="display:flex;gap:10px;flex-wrap:wrap">
        ${(dims.batting?.pp_specialists_zero_hp || []).slice(0, 3).map(r => `
          <span class="badge b-warn" onclick="openPlayer(${r.canonical_player_id})" style="cursor:pointer">
            ${r.player_name}: 0 Pressure Balls (${r.neff.toFixed(0)} Total Balls) — ${r.evidence_badge || 'No HP Evidence'}
          </span>
        `).join('')}
      </div>
    </div>

    <!-- Supporting Dimension A: Overall Multi-Phase Reliability (PRI 0–100) -->
    <div class="card-title" style="margin-top:28px">
      <span class="icon-title">
        <svg class="icon" style="color:var(--cyan)" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 14 14"></polyline></svg>
        <span>SUPPORTING DIMENSION A: OVERALL T20 RELIABILITY (ALL PHASES POOLED)</span>
      </span>
      <span class="kpi-sub">Evaluates volume consistency across ordinary overs (0–100 PRI scale)</span>
    </div>
    <div class="grid-2">
      <div class="card">
        <div class="card-title" style="font-size:0.92rem;color:var(--cyan)">Top 10 Batters (Overall Multi-Phase Consistency)</div>
        <div class="table-container"><table><thead><tr>
          <th>Rank</th><th>Player</th><th>${lbl('pri_lcb')}</th><th>Total Balls</th><th>Pressure Balls</th><th>HP Evidence</th>
        </tr></thead><tbody>
          ${(dims.batting?.top10_overall_reliability || []).map((r, i) => `
            <tr class="clickable" onclick="openPlayer(${r.canonical_player_id})">
              <td><strong>#${i+1}</strong></td>
              <td><div style="display:flex;align-items:center"><span class="player-avatar">${r.player_name.slice(0,2).toUpperCase()}</span><strong>${r.player_name}</strong></div></td>
              <td><strong style="color:var(--cyan)">${Number(r.pri_score_lcb).toFixed(2)}</strong></td>
              <td>${r.neff.toFixed(0)}</td>
              <td>${r.neff_hp != null ? r.neff_hp.toFixed(0) : '0'}</td>
              <td>${evidenceBadge(r.evidence_badge, r.neff_hp)}</td>
            </tr>
          `).join('')}
        </tbody></table></div>
      </div>

      <div class="card">
        <div class="card-title" style="font-size:0.92rem;color:var(--cyan)">Top 10 Bowlers (Overall Multi-Phase Consistency)</div>
        <div class="table-container"><table><thead><tr>
          <th>Rank</th><th>Player</th><th>${lbl('pri_lcb')}</th><th>Total Balls</th><th>Pressure Balls</th><th>HP Evidence</th>
        </tr></thead><tbody>
          ${(dims.bowling?.top10_overall_reliability || []).map((r, i) => `
            <tr class="clickable" onclick="openPlayer(${r.canonical_player_id})">
              <td><strong>#${i+1}</strong></td>
              <td><div style="display:flex;align-items:center"><span class="player-avatar">${r.player_name.slice(0,2).toUpperCase()}</span><strong>${r.player_name}</strong></div></td>
              <td><strong style="color:var(--cyan)">${Number(r.pri_score_lcb).toFixed(2)}</strong></td>
              <td>${r.neff.toFixed(0)}</td>
              <td>${r.neff_hp != null ? r.neff_hp.toFixed(0) : '0'}</td>
              <td>${evidenceBadge(r.evidence_badge, r.neff_hp)}</td>
            </tr>
          `).join('')}
        </tbody></table></div>
      </div>
    </div>
  `;
}

// ==========================================================================
// 2. BATTING SCOUTING
// ==========================================================================
async function renderBattingScouting(el) {
  const [d, dims] = await Promise.all([
    fetchJson('/api/dashboard-a?' + filterQuery({ discipline: 'batting' })),
    fetchJson('/api/evaluation-dimensions')
  ]);

  const batDims = dims.batting || {};

  el.innerHTML = `
    <div class="pill-tabs">
      <button class="pill ${STATE.batView==='hp_headline'?'active':''}" onclick="STATE.batView='hp_headline'; renderTab()">
        <svg class="icon" viewBox="0 0 24 24"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
        Top 10 High-Pressure Clutch Batters (Official Deliverable)
      </button>
      <button class="pill ${STATE.batView==='all'?'active':''}" onclick="STATE.batView='all'; renderTab()">
        <svg class="icon" viewBox="0 0 24 24"><line x1="8" y1="6" x2="21" y2="6"></line><line x1="8" y1="12" x2="21" y2="12"></line><line x1="8" y1="18" x2="21" y2="18"></line></svg>
        Complete Evaluated Universe (${d.total_eligible} Batters)
      </button>
    </div>

    ${STATE.batView === 'hp_headline' ? `
      <div class="card" style="border-top:3px solid var(--rr-pink);margin-bottom:24px">
        <div class="card-title" style="color:var(--rr-pink)">
          <span>Official Top 10 Batters Under Pressure</span>
          <span style="font-size:0.75rem;color:var(--text-dim);font-weight:400">Strictly filtered: Neff_HP &ge; 30</span>
        </div>
        <p class="kpi-sub" style="margin-top:-8px; margin-bottom:16px">
          These batters deliver the highest run production above contextual expectation during high-pressure chases (Pressure Index &ge; 60).
          <em>Lower-bound score (z = 0.5). Adjacent ranks are not statistically separable.</em>
        </p>
        <div class="grid-2">
          ${(batDims.top10_high_pressure_robust || []).map((r, i) => `
            <div class="scannable-row" onclick="openPlayer(${r.canonical_player_id})">
              <div class="scannable-row-main">
                <div class="scannable-row-left">
                  <div class="scannable-rank ${i < 3 ? 'rank-top3' : ''}">#${i+1}</div>
                  <div>
                    <div class="scannable-name-row">
                      <span class="scannable-name">${r.player_name}</span>
                      ${evidenceBadge(r.evidence_badge, r.neff_hp)}
                    </div>
                    <div class="scannable-score-sub">
                      HP-RAA (z=0.5): <strong>+${r.hp_score_lcb_runs_per_100b || r.hp_raa_lcb?.toFixed(2) || '—'}</strong> runs/100b &bull; ${Math.round(r.neff_hp || r.neff_hp_opportunities || 0)} balls
                    </div>
                  </div>
                </div>
                <div class="scannable-row-right">
                  <div class="scannable-metric-num">+${Number(r.hp_raa_post || r.hp_score_post_runs_per_100b || 0).toFixed(2)}</div>
                  <div class="scannable-metric-sub">mean r/100b</div>
                </div>
              </div>
              <div class="scannable-analyst-snippet">
                <strong>Analyst Verdict:</strong> ${r.analyst_blurb}
              </div>
            </div>
          `).join('')}
        </div>
      </div>
    ` : `
      ${filterBarHtml(true)}
      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th class="sortable" onclick="STATE.batSort = sortTableData(d.records, 'raw_pri_lcb_rank', true); renderTab()">Rank</th>
              <th>Player</th>
              <th>Role</th>
              <th class="sortable" onclick="STATE.batSort = sortTableData(d.records, 'pri_score_lcb', true)">PRI (LCB)</th>
              <th>90% CI</th>
              <th>Total Balls</th>
              <th class="sortable" onclick="STATE.batSort = sortTableData(d.records, 'neff_hp', true)">Pressure Balls</th>
              <th>HP Evidence Tier</th>
              <th>Availability</th>
            </tr>
          </thead>
          <tbody>
            ${d.records.map(r => `
              <tr class="clickable" onclick="openPlayer(${r.canonical_player_id})">
                <td><strong>#${r.raw_pri_lcb_rank}</strong></td>
                <td><div style="display:flex;align-items:center"><span class="player-avatar">${r.player_name.slice(0,2).toUpperCase()}</span><strong>${r.player_name}</strong></div></td>
                <td>${r.role_label}</td>
                <td><strong style="color:var(--cyan)">${Number(r.pri_score_lcb).toFixed(2)}</strong></td>
                <td>${r.rank_p05 && r.rank_p95 ? `[#${r.rank_p05}–#${r.rank_p95}]` : '—'}</td>
                <td>${Math.round(r.neff)}</td>
                <td><strong>${Math.round(r.neff_hp)}</strong></td>
                <td>${evidenceBadge(r.evidence_badge, r.neff_hp)}</td>
                <td>${statusBadge(r.availability_status)}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    `}
  `;
}

// ==========================================================================
// 3. BOWLING SCOUTING
// ==========================================================================
async function renderBowlingScouting(el) {
  const [d, dims] = await Promise.all([
    fetchJson('/api/dashboard-a?' + filterQuery({ discipline: 'bowling' })),
    fetchJson('/api/evaluation-dimensions')
  ]);

  const bowlDims = dims.bowling || {};

  el.innerHTML = `
    <div class="pill-tabs">
      <button class="pill ${STATE.bowlView==='hp_headline'?'active':''}" onclick="STATE.bowlView='hp_headline'; renderTab()">
        <svg class="icon" viewBox="0 0 24 24"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
        Top 10 High-Pressure Clutch Bowlers (Official Deliverable)
      </button>
      <button class="pill ${STATE.bowlView==='all'?'active':''}" onclick="STATE.bowlView='all'; renderTab()">
        <svg class="icon" viewBox="0 0 24 24"><line x1="8" y1="6" x2="21" y2="6"></line><line x1="8" y1="12" x2="21" y2="12"></line><line x1="8" y1="18" x2="21" y2="18"></line></svg>
        Complete Evaluated Universe (${d.total_eligible} Bowlers)
      </button>
    </div>

    ${STATE.bowlView === 'hp_headline' ? `
      <div class="card" style="border-top:3px solid var(--cyan);margin-bottom:24px">
        <div class="card-title" style="color:var(--cyan)">
          <span>Official Top 10 Bowlers Under Pressure</span>
          <span style="font-size:0.75rem;color:var(--text-dim);font-weight:400">Strictly filtered: Neff_HP &ge; 30</span>
        </div>
        <p class="kpi-sub" style="margin-top:-8px; margin-bottom:16px">
          Ranked by runs prevented per over above contextual expectation during high-pressure death overs (Pressure Index &ge; 60).
          <em>Lower-bound score (z = 0.5). Adjacent ranks are not statistically separable.</em>
        </p>
        <div class="grid-2">
          ${(bowlDims.top10_high_pressure_robust || []).map((r, i) => `
            <div class="scannable-row" onclick="openPlayer(${r.canonical_player_id})">
              <div class="scannable-row-main">
                <div class="scannable-row-left">
                  <div class="scannable-rank ${i < 3 ? 'rank-top3' : ''}" style="${i < 3 ? 'background:linear-gradient(135deg, var(--cyan) 0%, #254aa5 100%)' : ''}">#${i+1}</div>
                  <div>
                    <div class="scannable-name-row">
                      <span class="scannable-name">${r.player_name}</span>
                      ${evidenceBadge(r.evidence_badge, r.neff_hp)}
                    </div>
                    <div class="scannable-score-sub">
                      HP-Impact (z=0.5): <strong>+${r.hp_impact_lcb_runs_saved_per_ov || r.hp_impact_lcb?.toFixed(3) || '—'}</strong> r/ov &bull; ${Math.round(r.neff_hp || r.neff_hp_opportunities || 0)} balls
                    </div>
                  </div>
                </div>
                <div class="scannable-row-right">
                  <div class="scannable-metric-num" style="color:var(--cyan)">+${Number(r.hp_impact_post || r.hp_impact_post_runs_saved_per_ov || 0).toFixed(3)}</div>
                  <div class="scannable-metric-sub">mean r/ov saved</div>
                </div>
              </div>
              <div class="scannable-analyst-snippet" style="border-left-color:var(--cyan)">
                <strong>Analyst Verdict:</strong> ${r.analyst_blurb}
              </div>
            </div>
          `).join('')}
        </div>
      </div>
    ` : `
      ${filterBarHtml(true)}
      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Player</th>
              <th>Role</th>
              <th>PRI (LCB)</th>
              <th>90% CI</th>
              <th>Total Balls</th>
              <th>Pressure Balls</th>
              <th>HP Evidence Tier</th>
              <th>Availability</th>
            </tr>
          </thead>
          <tbody>
            ${d.records.map(r => `
              <tr class="clickable" onclick="openPlayer(${r.canonical_player_id})">
                <td><strong>#${r.raw_pri_lcb_rank}</strong></td>
                <td><div style="display:flex;align-items:center"><span class="player-avatar">${r.player_name.slice(0,2).toUpperCase()}</span><strong>${r.player_name}</strong></div></td>
                <td>${r.role_label}</td>
                <td><strong style="color:var(--cyan)">${Number(r.pri_score_lcb).toFixed(2)}</strong></td>
                <td>${r.rank_p05 && r.rank_p95 ? `[#${r.rank_p05}–#${r.rank_p95}]` : '—'}</td>
                <td>${Math.round(r.neff)}</td>
                <td><strong>${Math.round(r.neff_hp)}</strong></td>
                <td>${evidenceBadge(r.evidence_badge, r.neff_hp)}</td>
                <td>${statusBadge(r.availability_status)}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    `}
  `;
}

// ==========================================================================
// 4. HEAD-TO-HEAD PLAYER COMPARISON
// ==========================================================================
async function renderCompare(el) {
  const d = await fetchJson('/api/compare?ids=' + STATE.compareIds.join(','));

  el.innerHTML = `
    <div class="filter-panel" style="justify-content:space-between">
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
        <label class="filter-label">Compare IDs:</label>
        <input id="cmp-input" class="filter-input" type="text" value="${STATE.compareIds.join(',')}" style="width:240px" />
        <button class="pill active" onclick="STATE.compareIds=document.getElementById('cmp-input').value.split(',').map(x=>parseInt(x.trim())).filter(Boolean); renderTab()">
          Run Comparison
        </button>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <button class="pill" onclick="STATE.compareIds=[1699,1077,7260,2258]; renderTab()">Clutch Batters</button>
        <button class="pill" onclick="STATE.compareIds=[781,4641,4722,7920]; renderTab()">Clutch Bowlers</button>
        <button class="pill" onclick="STATE.compareIds=[50284,4847,1699,6396]; renderTab()">Powerplay vs Closers</button>
      </div>
    </div>

    <div class="table-container">
      <table>
        <thead>
          <tr>
            <th style="min-width:220px">Analytical Evaluation Metric</th>
            ${d.players.map(p => `
              <th>
                <div style="display:flex;align-items:center;gap:8px">
                  <span class="player-avatar">${p.player_name.slice(0,2).toUpperCase()}</span>
                  <div>
                    <div style="font-size:0.95rem;font-weight:700">${p.player_name}</div>
                    <div style="margin-top:2px">${statusBadge(p.availability_status)}</div>
                  </div>
                </div>
              </th>
            `).join('')}
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>2026 Team &rarr; 2027 Team</strong></td>
            ${d.players.map(p => `<td>${teamBadge(p.ipl_2026_team)} &rarr; ${teamBadge(p.current_2027_team)}</td>`).join('')}
          </tr>
          <tr>
            <td><strong>Nationality &amp; Cap Status</strong></td>
            ${d.players.map(p => `<td>${p.nationality} (${p.cap_status})</td>`).join('')}
          </tr>
          <tr>
            <td><strong>Batting Reliability (${lbl('pri_lcb')})</strong></td>
            ${d.players.map(p => `
              <td>
                ${p.batting ? `<strong style="color:var(--cyan)">${Number(p.batting.pri_score_lcb).toFixed(1)}</strong> <span class="kpi-sub">(#${p.batting.raw_pri_lcb_rank})</span>` : '—'}
              </td>
            `).join('')}
          </tr>
          <tr>
            <td><strong>High-Pressure Batting (${lbl('hp_raa')})</strong></td>
            ${d.players.map(p => `
              <td>
                ${p.batting ? (p.batting.neff_hp === 0 ? '<span class="chip-tier chip-no-hp">0 Pressure Balls</span>' : `<strong style="color:var(--green)">+${Number(p.batting.hp_raa_post).toFixed(2)}</strong> r/100b`) : '—'}
              </td>
            `).join('')}
          </tr>
          <tr>
            <td><strong>Bowling Reliability (${lbl('pri_lcb')})</strong></td>
            ${d.players.map(p => `
              <td>
                ${p.bowling ? `<strong style="color:var(--cyan)">${Number(p.bowling.pri_score_lcb).toFixed(1)}</strong> <span class="kpi-sub">(#${p.bowling.raw_pri_lcb_rank})</span>` : '—'}
              </td>
            `).join('')}
          </tr>
          <tr>
            <td><strong>High-Pressure Bowling (${lbl('hp_impact')})</strong></td>
            ${d.players.map(p => `
              <td>
                ${p.bowling ? `<strong style="color:var(--green)">+${Number(p.bowling.hp_impact_post).toFixed(3)}</strong> r/ov saved` : '—'}
              </td>
            `).join('')}
          </tr>
          <tr>
            <td><strong>Pressure Exposure (${lbl('neff_hp')})</strong></td>
            ${d.players.map(p => `
              <td>
                ${p.batting ? `Bat: <strong>${Math.round(p.batting.neff_hp)}</strong> ` : ''}
                ${p.bowling ? `Bowl: <strong>${Math.round(p.bowling.neff_hp)}</strong>` : ''}
              </td>
            `).join('')}
          </tr>
          <tr>
            <td><strong>Death Overs Dominance</strong></td>
            ${d.players.map(p => `
              <td>
                ${p.batting?.death_sr ? `Bat SR ${p.batting.death_sr}` : ''}
                ${p.bowling?.death_econ ? `Bowl Econ ${p.bowling.death_econ}` : '—'}
              </td>
            `).join('')}
          </tr>
          <tr>
            <td><strong>RR Acquisition Pathway</strong></td>
            ${d.players.map(p => `<td><span class="badge b-info">${p.acquisition_pathway}</span></td>`).join('')}
          </tr>
        </tbody>
      </table>
    </div>
  `;
}

// ==========================================================================
// 5. PRESSURE & WIN IMPACT (WPA & LEVERAGE)
// ==========================================================================
async function renderPressureLeverage(el) {
  const wp = await fetchJson('/api/win-probability');

  el.innerHTML = `
    <div class="alert alert-info">
      <div>
        <strong style="color:var(--cyan)">Empirical Orthogonality Proof:</strong>
        Pressure Index (PI, 0–100) measures pre-delivery situational urgency and difficulty.
        Leverage Index (LI) measures counterfactual win-probability swing ($|P(\\text{Win}|\\text{Six}) - P(\\text{Win}|\\text{Wicket})|$).
        <strong>Overall correlation: Pearson r = +0.0058, Spearman &rho; = -0.0185.</strong>
        In Innings 1, r = -0.37 (early collapses swing win prob without high PI).
        In Innings 2, r = +0.08 (lost chases feature extreme RRR with near-zero win swing).
      </div>
    </div>

    <div class="grid-4">
      <div class="kpi-card" style="--accent-color:var(--cyan)">
        <div class="kpi-label">Average Outcome Swing</div>
        <div class="kpi-val" style="color:var(--cyan)">${wp.mean_leverage} pp</div>
        <div class="kpi-sub">Win probability shift per delivery</div>
      </div>
      <div class="kpi-card" style="--accent-color:var(--purple)">
        <div class="kpi-label">High-Leverage Cutoff</div>
        <div class="kpi-val">${wp.p90_leverage} pp</div>
        <div class="kpi-sub">Top 10% match-defining moments</div>
      </div>
      <div class="kpi-card" style="--accent-color:var(--gold)">
        <div class="kpi-label">Orthogonal Correlation</div>
        <div class="kpi-val" style="color:var(--gold)">r = +0.01</div>
        <div class="kpi-sub">Distinct mathematical dimensions</div>
      </div>
      <div class="kpi-card" style="--accent-color:var(--green)">
        <div class="kpi-label">Evaluated Match States</div>
        <div class="kpi-val" style="color:var(--green)">182,311</div>
        <div class="kpi-sub">Decided match deliveries</div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-title" style="color:var(--cyan)">Top 10 Batters by Win Probability Added (WPA)</div>
        <div class="table-container">
          <table>
            <thead><tr><th>Player</th><th>Total WPA</th><th>WPA / 100 Balls</th><th>High-Leverage Balls</th></tr></thead>
            <tbody>
              ${(wp.top_bat_wpa || []).map(r => `
                <tr class="clickable" onclick="openPlayer(${r.canonical_striker_id})">
                  <td><div style="display:flex;align-items:center"><span class="player-avatar">${r.player_name.slice(0,2).toUpperCase()}</span><strong>${r.player_name}</strong></div></td>
                  <td><strong style="color:var(--green)">+${Number(r.total_wpa).toFixed(2)}</strong></td>
                  <td>+${Number(r.wpa_per_100).toFixed(2)}%</td>
                  <td>${r.hl_balls}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <div class="card-title" style="color:var(--rr-pink)">Top 10 Bowlers by Win Probability Added (WPA)</div>
        <div class="table-container">
          <table>
            <thead><tr><th>Player</th><th>Total WPA</th><th>WPA / 100 Balls</th><th>High-Leverage Balls</th></tr></thead>
            <tbody>
              ${(wp.top_bowl_wpa || []).map(r => `
                <tr class="clickable" onclick="openPlayer(${r.canonical_bowler_id})">
                  <td><div style="display:flex;align-items:center"><span class="player-avatar">${r.player_name.slice(0,2).toUpperCase()}</span><strong>${r.player_name}</strong></div></td>
                  <td><strong style="color:var(--green)">+${Number(r.total_wpa).toFixed(2)}</strong></td>
                  <td>+${Number(r.wpa_per_100).toFixed(2)}%</td>
                  <td>${r.hl_balls}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  `;
}

// ==========================================================================
// 6. TACTICAL CRICKET INTELLIGENCE
// ==========================================================================
async function renderCricketIntelligence(el) {
  const data = await fetchJson('/api/cricket-intelligence');
  const entries = STATE.intelView === 'profiles'
    ? Object.entries(data.strategic_profiles || {}).map(([id, row]) => ({ id: Number(id), name: row.player_name || '', ...row }))
    : Object.values((STATE.intelView === 'batting' ? data.batting_intelligence : data.bowling_intelligence) || {}).map(row => ({ ...row, id: row.canonical_player_id, name: row.player_name }));

  const query = STATE.intelSearch.trim().toLowerCase();
  const rows = entries.filter(row => `${row.name} ${row.tactical_archetype || ''} ${row.composure_tier || ''} ${row.tactical_role || ''}`.toLowerCase().includes(query));
  const isBat = STATE.intelView === 'batting';
  const profileView = STATE.intelView === 'profiles';

  el.innerHTML = `
    <div class="pill-tabs">
      <button class="pill ${STATE.intelView==='batting'?'active':''}" onclick="STATE.intelView='batting'; renderTab()">Batting Responses</button>
      <button class="pill ${STATE.intelView==='bowling'?'active':''}" onclick="STATE.intelView='bowling'; renderTab()">Bowling Responses</button>
      <button class="pill ${profileView?'active':''}" onclick="STATE.intelView='profiles'; renderTab()">Strategic Profiles</button>
    </div>

    <div class="filter-panel">
      <input class="filter-input" type="search" placeholder="Search player or archetype..." value="${STATE.intelSearch.replace(/"/g, '&quot;')}" oninput="STATE.intelSearch=this.value; renderTab()" style="width:280px" />
      <span class="kpi-sub">${rows.length} tactical records</span>
    </div>

    <div class="table-container">
      <table>
        <thead>
          <tr>
            ${profileView
              ? '<th>Player</th><th>Discovery</th><th>Significance</th><th>RR Implication</th><th>Scouting Decision</th>'
              : isBat
                ? '<th>Player</th><th>Archetype</th><th>HP Balls</th><th>vs Pace SR</th><th>vs Spin SR</th><th>Post-Dot Recovery</th><th>Composure</th>'
                : '<th>Player</th><th>Tactical Role</th><th>Post-Boundary Opps</th><th>Economy</th><th>Repeat Boundary %</th><th>Dot Rebound %</th><th>Bounceback</th>'}
          </tr>
        </thead>
        <tbody>
          ${rows.map(row => profileView
            ? `<tr class="clickable" onclick="openPlayer(${row.id})">
                 <td><strong>${row.name}</strong></td>
                 <td>${row.q1_discovery || '—'}</td>
                 <td>${row.q2_significance || '—'}</td>
                 <td>${row.q3_rr_implication || '—'}</td>
                 <td><strong style="color:var(--cyan)">${row.q4_scouting_decision || '—'}</strong></td>
               </tr>`
            : isBat
              ? `<tr class="clickable" onclick="openPlayer(${row.id})">
                   <td><strong>${row.name}</strong></td>
                   <td>${row.tactical_archetype || '—'}</td>
                   <td><strong>${row.hp_balls ?? '—'}</strong></td>
                   <td>${row.hp_pace_sr ? row.hp_pace_sr + ' SR' : '—'}</td>
                   <td>${row.hp_spin_sr ? row.hp_spin_sr + ' SR' : '—'}</td>
                   <td>${row.post_dot_sr ? row.post_dot_sr + ' SR (' + row.post_dot_opps + ' opps)' : '—'}</td>
                   <td>${evidenceBadge(row.composure_tier)}</td>
                 </tr>`
              : `<tr class="clickable" onclick="openPlayer(${row.id})">
                   <td><strong>${row.name}</strong></td>
                   <td>${row.tactical_role || '—'}</td>
                   <td>${row.post_b_opps ?? '—'} opps</td>
                   <td><strong style="color:var(--green)">${row.post_b_econ ?? '—'}</strong></td>
                   <td>${row.post_b_repeat_b_pct != null ? row.post_b_repeat_b_pct + '%' : '—'}</td>
                   <td>${row.post_b_dot_pct != null ? row.post_b_dot_pct + '%' : '—'}</td>
                   <td>${evidenceBadge(row.bounceback_tier)}</td>
                 </tr>`
          ).join('')}
        </tbody>
      </table>
    </div>
  `;
}

// ==========================================================================
// 7. AUCTION INTELLIGENCE & ACQUISITION BUCKETS (3-BLOCK CARDS)
// ==========================================================================
async function renderAuctionIntelligence(el) {
  const [intel, d] = await Promise.all([
    fetchJson('/api/rr-tactical-fit'),
    fetchJson('/api/dashboard-b?' + filterQuery())
  ]);

  const buckets = [
    ['pending_retention_watchlist', 'Pending Retention Watchlist'],
    ['actionable_auction_targets', 'Actionable Auction Targets'],
    ['confirmed_released_pipeline', 'Confirmed Released Pipeline'],
    ['rr_2026_squad_planning', 'RR 2026 Squad Planning'],
    ['confirmed_other_team_excluded', 'Confirmed Other Team (Excluded)'],
    ['retired_unavailable_excluded', 'Retired / Unavailable (Excluded)'],
  ];

  const validBuckets = buckets.map(b => b[0]);
  if (!validBuckets.includes(STATE.auctionBucket)) {
    STATE.auctionBucket = 'pending_retention_watchlist';
  }

  const rows = d.buckets[STATE.auctionBucket] || [];

  el.innerHTML = `
    <div class="alert alert-info">
      <div>
        <strong style="color:var(--cyan)">BCCI IPL 2027 Auction Governance:</strong>
        Retention deadline: <strong>November 17, 2026</strong>; auction registrations close <strong>November 23, 2026</strong>. Mini-auction: <strong>December 16, 2026 in Goa</strong>.
        <em>Strict Compliance Rule: Only formally released or registered players are labeled "Target"; all others are "Watchlist".</em>
      </div>
    </div>

    <div class="card-title">Priority Tactical Gaps &amp; Target Shortlists</div>
    <div class="grid-3">
      ${intel.identified_roster_gaps.map(g => `
        <div class="card" style="border-top:3px solid ${g.priority==='HIGH'?'var(--rr-pink)':'var(--gold)'}">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
            <strong style="font-size:1.0rem">${g.gap}</strong>
            <span class="badge ${g.priority==='HIGH'?'b-warn':'b-pending'}">${g.priority} PRIORITY</span>
          </div>
          <p class="kpi-sub" style="margin-bottom:14px">${g.reason}</p>
          <div style="font-size:0.72rem;text-transform:uppercase;color:var(--text-dim);font-weight:700;margin-bottom:8px">Shortlisted Candidates:</div>
          ${g.target_candidates.map(c => `
            <div style="background:var(--bg-surface);border:1px solid var(--border);border-radius:var(--radius-sm);padding:8px 12px;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center">
              <div>
                <strong>${c.name}</strong> <span class="kpi-sub">(${c.team})</span>
              </div>
              <div style="display:flex;align-items:center;gap:6px">
                <span class="badge b-info">${c.pathway}</span>
                <strong style="color:var(--cyan)">${c.pri}</strong>
              </div>
            </div>
          `).join('')}
        </div>
      `).join('')}
    </div>

    <div class="card-title" style="margin-top:28px">Auction Eligibility Buckets</div>
    <div class="pill-tabs">
      ${buckets.map(([k, lblText]) => `
        <button class="pill ${STATE.auctionBucket===k?'active':''}" onclick="STATE.auctionBucket='${k}'; renderTab()">
          ${lblText} (${d.counts[k] || 0})
        </button>
      `).join('')}
    </div>

    <!-- 3-Block Candidate Presentation -->
    <div style="margin-top:16px">
      ${rows.slice(0, 15).map(r => `
        <div class="scouting-card-3block">
          <div class="scouting-card-header">
            <div style="display:flex;align-items:center;gap:12px">
              <span class="player-avatar" style="width:36px;height:36px;font-size:0.9rem">${r.player_name.slice(0,2).toUpperCase()}</span>
              <div>
                <div style="font-size:1.05rem;font-weight:700">${r.player_name}</div>
                <div class="kpi-sub">${r.role_label} &bull; ${r.discipline}</div>
              </div>
            </div>
            <div style="display:flex;align-items:center;gap:8px">
              ${statusBadge(r.availability_status)}
              <button class="pill" onclick="openPlayer(${r.canonical_player_id})">Full Profile &rarr;</button>
            </div>
          </div>
          <div class="scouting-blocks-grid">
            <div class="scouting-block">
              <div class="scouting-block-title block-perf">
                <svg class="icon" viewBox="0 0 24 24" style="width:14px;height:14px"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
                1. Performance Evidence
              </div>
              <div class="kv-grid" style="margin:0">
                <div class="kv"><div class="k">Reliability (PRI)</div><div class="v">${r.pri_score_lcb}</div></div>
                <div class="kv"><div class="k">Pressure Balls</div><div class="v">${r.neff_hp}</div></div>
              </div>
            </div>
            <div class="scouting-block">
              <div class="scouting-block-title block-tact">
                <svg class="icon" viewBox="0 0 24 24" style="width:14px;height:14px"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 14 14"></polyline></svg>
                2. Tactical Suitability
              </div>
              <div class="kpi-sub" style="font-size:0.82rem;line-height:1.4">
                Fit for SMS Jaipur pace void or Guwahati spin track. Evaluated for specific late-overs execution.
              </div>
            </div>
            <div class="scouting-block">
              <div class="scouting-block-title block-avail">
                <svg class="icon" viewBox="0 0 24 24" style="width:14px;height:14px"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path></svg>
                3. Availability Status
              </div>
              <div style="font-size:0.84rem;font-weight:600;margin-bottom:4px">${teamBadge(r.current_2027_team)}</div>
              <div class="kpi-sub">Pathway: <strong style="color:var(--text-primary)">${r.acquisition_pathway}</strong></div>
            </div>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

// ==========================================================================
// 8. RAJASTHAN ROYALS SQUAD 2026 AUDIT
// ==========================================================================
async function renderRRSquadAudit(el) {
  const squad = await fetchJson('/api/rr-squad');

  el.innerHTML = `
    <div class="alert alert-info">
      <div>
        <strong style="color:var(--rr-pink)">Rajasthan Royals 2026 Performance Audit:</strong>
        RR finished <strong>8–6 in the regular league (4th place)</strong> and <strong>9–7 overall (3rd place)</strong>.
        Venue split analysis shows RR was <strong>3/3 at ACA Guwahati</strong> (spin equilibrium), but struggled <strong>1/4 at SMS Jaipur</strong> due to a 10.13 RPO pace economy leak in death overs.
      </div>
    </div>

    <div class="grid-4">
      <div class="kpi-card" style="--accent-color:var(--rr-pink)">
        <div class="kpi-label">Guwahati Record</div>
        <div class="kpi-val" style="color:var(--green)">3 – 0 (100%)</div>
        <div class="kpi-sub">Spin dominance</div>
      </div>
      <div class="kpi-card" style="--accent-color:var(--red)">
        <div class="kpi-label">SMS Jaipur Record</div>
        <div class="kpi-val" style="color:var(--red)">1 – 4 (20%)</div>
        <div class="kpi-sub">10.13 RPO pace leak</div>
      </div>
      <div class="kpi-card" style="--accent-color:var(--gold)">
        <div class="kpi-label">Death Batting SR</div>
        <div class="kpi-val" style="color:var(--gold)">163.4</div>
        <div class="kpi-sub">League average: 169.4</div>
      </div>
      <div class="kpi-card" style="--accent-color:var(--cyan)">
        <div class="kpi-label">Audited Roster</div>
        <div class="kpi-val">${squad.roster.length} Players</div>
        <div class="kpi-sub">Retained &amp; reviewed</div>
      </div>
    </div>

    <div class="card-title">Retained Roster Performance Audit</div>
    <div class="table-container">
      <table>
        <thead>
          <tr>
            <th>Player</th>
            <th>2026 Apps</th>
            <th>Bat Order</th>
            <th>Batting PRI</th>
            <th>Bat HP Balls</th>
            <th>Bowling PRI</th>
            <th>Bowl HP Balls</th>
          </tr>
        </thead>
        <tbody>
          ${squad.roster.map(r => `
            <tr class="clickable" onclick="openPlayer(${r.canonical_player_id})">
              <td><div style="display:flex;align-items:center"><span class="player-avatar">${r.canonical_name.slice(0,2).toUpperCase()}</span><strong>${r.canonical_name}</strong></div></td>
              <td>${r.apps_2026} matches</td>
              <td>${r.med_bat_order ?? '—'}</td>
              <td>${r.bat_pri_lcb ? `<strong style="color:var(--cyan)">${r.bat_pri_lcb.toFixed(1)}</strong>` : '—'}</td>
              <td>${r.bat_neff_hp != null ? (r.bat_neff_hp === 0 ? '<span class="chip-tier chip-no-hp">0</span>' : Math.round(r.bat_neff_hp)) : '—'}</td>
              <td>${r.bowl_pri_lcb ? `<strong style="color:var(--rr-pink)">${r.bowl_pri_lcb.toFixed(1)}</strong>` : '—'}</td>
              <td>${r.bowl_neff_hp != null ? Math.round(r.bowl_neff_hp) : '—'}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>
  `;
}

// ==========================================================================
// 9. IPL 2027 TRADE TRACKER
// ==========================================================================
async function renderTrades(el) {
  const trades = await fetchJson('/api/trades');

  el.innerHTML = `
    <div class="alert alert-info">
      <div>
        <strong style="color:var(--cyan)">IPL 2027 Verified Trade Ledger:</strong>
        Four official multi-player trades have been formally registered by BCCI as of <strong>${trades.metadata.as_of_date}</strong>.
        All contract transfers and adjusted salaries have been reflected in player profiles.
      </div>
    </div>

    <div class="card-title">Officially Completed Franchise Trades (4 Verified)</div>
    <div class="grid-2">
      ${trades.official_completed_trades.map(t => `
        <div class="card" style="margin-bottom:12px">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
            <span class="badge badge-target">${t.transaction_status}</span>
            <span class="kpi-sub font-mono">${t.date} &bull; ${t.trade_id}</span>
          </div>
          <div>
            ${t.movements.map(m => `
              <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid var(--border-subtle)">
                <div>
                  <strong style="cursor:pointer;color:var(--cyan)" onclick="openPlayer(${m.canonical_player_id})">${m.player_name}</strong>
                  <span class="kpi-sub">(${teamBadge(m.from_team)} &rarr; ${teamBadge(m.to_team)})</span>
                </div>
                <div style="text-align:right">
                  <span class="font-mono" style="color:var(--green);font-weight:700">INR ${m.fee_cr} Cr</span>
                </div>
              </div>
            `).join('')}
          </div>
        </div>
      `).join('')}
    </div>

    <div class="card-title" style="margin-top:24px">Reported Negotiations &amp; Stalled Rumours</div>
    ${trades.reported_negotiations_not_completed.map(r => `
      <div class="card" style="border-left:4px solid var(--gold);margin-bottom:12px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
          <strong style="color:var(--gold)">${r.transaction_status} (${r.rumour_id})</strong>
          <span class="kpi-sub font-mono">${r.date}</span>
        </div>
        <p style="font-size:0.86rem;margin-bottom:10px">${r.summary}</p>
        <div style="display:flex;gap:8px;flex-wrap:wrap">
          ${r.players_mentioned.map(p => `
            <span class="badge badge-watchlist">${p.player_name}: ${teamBadge(p.current_team)} (interest from ${teamBadge(p.reported_interest_from)})</span>
          `).join('')}
        </div>
      </div>
    `).join('')}
  `;
}

// ==========================================================================
// 10. MODEL VALIDATION & SENSITIVITY BENCHMARKS
// ==========================================================================
async function renderModelEvaluation(el) {
  const [wp, ws] = await Promise.all([
    fetchJson('/api/win-probability'),
    fetchJson('/api/weight-sensitivity')
  ]);

  const models = wp?.overall_metrics || {};
  const batSens = ws?.batting || {};
  const bowlSens = ws?.bowling || {};

  el.innerHTML = `
    <div class="card-title">1. Win Probability Benchmark (5-Fold Match-Grouped GroupKFold)</div>
    <p class="kpi-sub" style="margin-top:-8px; margin-bottom:16px">
      Evaluated strictly on pre-delivery match state without outcome leakage. Cubic splines logistic regression achieved the top calibration score (0.7785 overall, 0.8962 in chases).
    </p>
    <div class="table-container" style="margin-bottom:24px">
      <table>
        <thead>
          <tr><th>Architecture</th><th>ROC-AUC</th><th>PR-AUC</th><th>Brier Score</th><th>Log Loss</th><th>Mean Calibration Error</th></tr>
        </thead>
        <tbody>
          ${Object.entries(models).map(([name, m]) => `
            <tr>
              <td><strong>${name.replace('_', ' ').toUpperCase()}</strong></td>
              <td><strong style="color:var(--cyan)">${m.roc_auc}</strong></td>
              <td>${m.pr_auc}</td>
              <td>${m.brier_score}</td>
              <td>${m.log_loss}</td>
              <td><strong style="color:var(--green)">${m.mean_calibration_error}</strong></td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>

    <div class="card-title">2. Tactical Weight Sensitivity (4-Scenario Correlation Matrix)</div>
    <div class="grid-2" style="margin-bottom:24px">
      <div class="card">
        <div class="card-title" style="font-size:0.92rem">Batting Correlation Matrix (188 Eligible)</div>
        <div class="table-container">
          <table>
            <thead><tr><th>Scenario</th><th>Baseline</th><th>Balanced</th><th>Pressure</th><th>Reliability</th></tr></thead>
            <tbody>
              ${Object.entries(batSens.correlation_matrix || {}).map(([s1, row]) => `
                <tr>
                  <td><strong>${s1}</strong></td>
                  <td><strong style="color:var(--cyan)">${row.Baseline}</strong></td>
                  <td>${row.Balanced}</td>
                  <td>${row['Pressure-Focused']}</td>
                  <td>${row['Reliability-Focused']}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <div class="card-title" style="font-size:0.92rem">Bowling Correlation Matrix (200 Eligible)</div>
        <div class="table-container">
          <table>
            <thead><tr><th>Scenario</th><th>Baseline</th><th>Balanced</th><th>Pressure</th><th>Reliability</th></tr></thead>
            <tbody>
              ${Object.entries(bowlSens.correlation_matrix || {}).map(([s1, row]) => `
                <tr>
                  <td><strong>${s1}</strong></td>
                  <td><strong style="color:var(--cyan)">${row.Baseline}</strong></td>
                  <td>${row.Balanced}</td>
                  <td>${row['Pressure-Focused']}</td>
                  <td>${row['Reliability-Focused']}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  `;
}

// ==========================================================================
// 11. TECHNICAL METHODOLOGY & FORMULAS
// ==========================================================================
async function renderMethodology(el) {
  const m = await fetchJson('/api/methodology');
  const acc = m.accounting;

  el.innerHTML = `
    <div class="card-title">Data Accounting &amp; Integrity Audit</div>
    <div class="grid-4">
      <div class="kpi-card" style="--accent-color:var(--cyan)">
        <div class="kpi-label">Raw Matches</div>
        <div class="kpi-val">${acc.n_raw_matches}</div>
        <div class="kpi-sub">${acc.n_matches_with_bbb} with ball-by-ball</div>
      </div>
      <div class="kpi-card" style="--accent-color:var(--purple)">
        <div class="kpi-label">Raw Deliveries</div>
        <div class="kpi-val">${acc.n_raw_deliveries.toLocaleString()}</div>
        <div class="kpi-sub">${acc.n_regular_deliveries.toLocaleString()} regular / ${acc.n_super_over_deliveries} SO</div>
      </div>
      <div class="kpi-card" style="--accent-color:var(--gold)">
        <div class="kpi-label">DLS Suspected Excluded</div>
        <div class="kpi-val" style="color:var(--gold)">${acc.n_suspected_dls_excluded}</div>
        <div class="kpi-sub">Rain truncation integrity guard</div>
      </div>
      <div class="kpi-card" style="--accent-color:var(--green)">
        <div class="kpi-label">Emerging Asia Cup</div>
        <div class="kpi-val">${acc.n_emerging_total.toLocaleString()}</div>
        <div class="kpi-sub">Evaluated as separate track</div>
      </div>
    </div>

    <div class="card" style="margin-top:20px">
      <div class="card-title">Mathematical &amp; Statistical Formulas</div>
      <div class="kv-grid">
        <div class="kv">
          <div class="k">1. Zero-Leakage Pre-Delivery Game State</div>
          <div class="v" style="font-size:0.84rem;font-weight:400">All pressure metrics, required run rates, and win probabilities are computed strictly BEFORE delivery outcome is revealed.</div>
        </div>
        <div class="kv">
          <div class="k">2. Kish Effective Sample Size (Neff)</div>
          <div class="v" style="font-size:0.84rem;font-weight:400">Neff = (Sum w_i)^2 / Sum(w_i^2) with IPL deliveries receiving w=1.0 and T20I deliveries receiving w=0.5.</div>
        </div>
        <div class="kv">
          <div class="k">3. Empirical Bayesian Shrinkage</div>
          <div class="v" style="font-size:0.84rem;font-weight:400">Shrinks observed rates toward population means inversely proportional to sample size: theta_post = (Neff * theta_obs + M * theta_prior) / (Neff + M).</div>
        </div>
        <div class="kv">
          <div class="k">4. Lower Credible Bound (LCB z = 0.5)</div>
          <div class="v" style="font-size:0.84rem;font-weight:400">Score = theta_post - 0.50 * SD. Conservative lower bound (30.85th percentile), not a 95% confidence interval or probability.</div>
        </div>
      </div>
    </div>

    <div class="card" style="margin-top:20px;border-left:4px solid var(--red)">
      <div class="card-title" style="color:var(--red)">Explicit Data Limitations &amp; Boundaries</div>
      <div class="kpi-sub" style="font-size:0.88rem;line-height:1.6">
        1. <strong>Competition Scope:</strong> Evaluates 2022–2026 IPL and T20I fixtures. Domestic leagues (SMA, PSL, BBL, CPL) excluded to prevent cross-league calibration distortion.<br>
        2. <strong>Small High-Pressure Sample:</strong> Only ~11% of deliveries occur under severe pressure (PI &ge; 60). Players with Neff_HP &lt; 30 are excluded from official high-pressure lists.<br>
        3. <strong>Hawkeye Telemetry Absence:</strong> The dataset lacks ball-tracking pitch coordinates, meaning ball swing, seam movement, and release height cannot be modeled directly.<br>
        4. <strong>Age Field Absence:</strong> Bio age is compiled from external verified registries rather than raw ball-by-ball scorecards.
      </div>
    </div>
  `;
}

// ==========================================================================
// 12. PLAYER DETAIL DRAWER (MODAL)
// ==========================================================================
async function openPlayer(cid) {
  let p, dims;
  try {
    [p, dims] = await Promise.all([
      fetchJson('/api/player/' + cid),
      fetchJson('/api/evaluation-dimensions')
    ]);
  } catch (err) {
    console.error('Failed to load player modal data:', err);
    return;
  }
  if (!p) return;

  let analystBlurb = "Evaluated under standard IPL and T20I competition conditions.";
  let evBadge = "Moderate Evidence";
  const allBat = [...(dims.batting?.top10_high_pressure_robust || []), ...(dims.batting?.top10_overall_reliability || []), ...(dims.batting?.pp_specialists_zero_hp || [])];
  const allBowl = [...(dims.bowling?.top10_high_pressure_robust || []), ...(dims.bowling?.top10_overall_reliability || [])];
  const found = allBat.find(x => x.canonical_player_id === cid) || allBowl.find(x => x.canonical_player_id === cid);
  if (found) {
    if (found.analyst_blurb) analystBlurb = found.analyst_blurb;
    if (found.evidence_badge) evBadge = found.evidence_badge;
  }

  const discBlock = (label, d, isBat) => {
    if (!d) return '';
    return `
      <div class="card" style="margin-bottom:16px">
        <div class="card-title" style="margin-bottom:8px">
          <span class="icon-title">
            <svg class="icon" style="color:${isBat?'var(--cyan)':'var(--rr-pink)'}" viewBox="0 0 24 24"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
            ${label} — Rank #${d.raw_pri_lcb_rank} (${d.emp_pct_elig_primary}th percentile)
          </span>
          <span class="font-mono">${lbl('pri_lcb')}: <strong style="color:var(--cyan)">${Number(d.pri_score_lcb).toFixed(1)}</strong></span>
        </div>
        ${d.hp_banner ? `<div class="alert alert-warn" style="margin-bottom:12px"><strong>Pressure Exposure Alert:</strong> ${d.hp_banner}</div>` : ''}
        <div class="kv-grid">
          <div class="kv"><div class="k">Total Balls / Pressure Balls</div><div class="v">${Math.round(d.neff)} / <strong>${Math.round(d.neff_hp)}</strong></div></div>
          <div class="kv"><div class="k">Ranking 90% CI</div><div class="v">${d.rank_p05 && d.rank_p95 ? `#${d.rank_p05}–#${d.rank_p95} (${d.top10_inc_pct}% Top-10)` : '—'}</div></div>
          <div class="kv"><div class="k">Delivery Volume</div><div class="v">${d.ipl_balls} IPL (${d.ipl_2026_balls} in '26) / ${d.t20i_balls} T20I</div></div>
          <div class="kv"><div class="k">Tactical Role / Hand</div><div class="v">${d.role_label} (${isBat ? d.batting_hand : d.bowling_subtype})</div></div>
          ${isBat ? `
            <div class="kv"><div class="k">Runs Above Expected (Mean)</div><div class="v">${d.raa_post} runs/100b</div></div>
            <div class="kv"><div class="k">High-Pressure RAA (z=0.5)</div><div class="v"><strong style="color:var(--green)">+${d.hp_raa_post} r/100b</strong></div></div>
            <div class="kv"><div class="k">True Strike Rate</div><div class="v">${d.true_sr ?? '—'} (IPL: ${d.ipl_true_sr ?? '—'})</div></div>
            <div class="kv"><div class="k">Phase SR (PP/Mid/Death)</div><div class="v">${d.powerplay_sr ?? '—'} / ${d.middle_sr ?? '—'} / ${d.death_sr ?? '—'}</div></div>
          ` : `
            <div class="kv"><div class="k">Runs Saved (Mean)</div><div class="v">${d.rsaa_post} r/ov</div></div>
            <div class="kv"><div class="k">High-Pressure Impact (z=0.5)</div><div class="v"><strong style="color:var(--green)">+${d.hp_impact_post} r/ov</strong></div></div>
            <div class="kv"><div class="k">True Economy</div><div class="v">${d.true_econ ?? '—'} (IPL: ${d.ipl_true_econ ?? '—'})</div></div>
            <div class="kv"><div class="k">Phase Econ (PP/Mid/Death)</div><div class="v">${d.powerplay_econ ?? '—'} / ${d.middle_econ ?? '—'} / ${d.death_econ ?? '—'}</div></div>
          `}
        </div>
      </div>
    `;
  };

  const stratProf = p.batting?.strategic_profile || p.bowling?.strategic_profile;

  document.getElementById('player-modal-body').innerHTML = `
    <div class="modal-head">
      <div style="display:flex;align-items:center;gap:14px">
        <span class="player-avatar" style="width:44px;height:44px;font-size:1.1rem">${p.player_name.slice(0,2).toUpperCase()}</span>
        <div>
          <h2 style="font-size:1.3rem;margin-bottom:4px">${p.player_name} <span class="font-mono" style="font-size:0.8rem;color:var(--text-dim)">(ID: ${p.canonical_player_id})</span></h2>
          <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center">
            ${statusBadge(p.availability_status)}
            ${evidenceBadge(evBadge, p.batting?.neff_hp ?? p.bowling?.neff_hp)}
            <span class="kpi-sub">2026: ${teamBadge(p.ipl_2026_team)} &rarr; 2027: ${teamBadge(p.current_2027_team)}</span>
            <span class="kpi-sub">| ${p.nationality} | ${p.cap_status} | Age: ${p.age_2026}</span>
          </div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:8px">
        <button class="pill" onclick="toggleQuickCompare(${p.canonical_player_id}, '${p.player_name.replace(/'/g, "\\'")}'); closeModal();">
          ${(STATE.quickCompare && STATE.quickCompare.some(x => x.id === p.canonical_player_id)) ? 'In Compare Tray' : 'Compare'}
        </button>
        <button class="btn-close" onclick="closeModal()">
          <svg class="icon" viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
        </button>
      </div>
    </div>

    <div class="modal-body">
      <!-- 3-Block Candidate Structure: Performance, Tactical, Availability -->
      <div class="scouting-blocks-grid" style="margin-bottom:16px">
        <div class="scouting-block">
          <div class="scouting-block-title block-perf">1. Performance Evidence</div>
          <div class="kpi-sub" style="font-size:0.85rem">
            ${p.batting ? `Batting LCB: <strong>${p.batting.pri_score_lcb}</strong> (#${p.batting.raw_pri_lcb_rank})<br>` : ''}
            ${p.bowling ? `Bowling LCB: <strong>${p.bowling.pri_score_lcb}</strong> (#${p.bowling.raw_pri_lcb_rank})<br>` : ''}
            HP Opportunities: <strong>${Math.round(p.batting?.neff_hp ?? p.bowling?.neff_hp ?? 0)} balls</strong>
          </div>
        </div>
        <div class="scouting-block">
          <div class="scouting-block-title block-tact">2. Tactical Suitability</div>
          <div class="kpi-sub" style="font-size:0.85rem">
            ${p.batting ? `Role: <strong>${p.batting.role_label}</strong><br>Death SR: <strong>${p.batting.death_sr ?? '—'}</strong>` : ''}
            ${p.bowling ? `Role: <strong>${p.bowling.role_label}</strong><br>Death Econ: <strong>${p.bowling.death_econ ?? '—'}</strong>` : ''}
          </div>
        </div>
        <div class="scouting-block">
          <div class="scouting-block-title block-avail">3. Availability Status</div>
          <div class="kpi-sub" style="font-size:0.85rem">
            Status: ${statusBadge(p.availability_status)}<br>
            Pathway: <strong style="color:var(--text-primary)">${p.acquisition_pathway}</strong>
          </div>
        </div>
      </div>

      <!-- Prominent Analyst Verdict Box -->
      <div class="scannable-analyst-snippet" style="margin-bottom:16px;border-left:4px solid var(--rr-pink)">
        <div style="font-size:0.75rem;text-transform:uppercase;color:var(--rr-pink);font-weight:700;margin-bottom:4px">Cricket Analyst Scouting Assessment:</div>
        <div style="font-size:0.92rem;color:var(--text-primary)">${analystBlurb}</div>
      </div>

      ${stratProf ? `
        <div class="card" style="margin-bottom:16px;border-left:4px solid var(--gold)">
          <div style="font-size:0.75rem;text-transform:uppercase;color:var(--gold);font-weight:700;margin-bottom:6px">
            Rajasthan Royals 4-Question Strategic Intelligence:
          </div>
          <div style="font-size:0.84rem;margin-bottom:5px"><strong>1. Discovery:</strong> ${stratProf.q1_discovery}</div>
          <div style="font-size:0.84rem;margin-bottom:5px"><strong>2. Significance:</strong> ${stratProf.q2_significance}</div>
          <div style="font-size:0.84rem;margin-bottom:5px"><strong>3. RR Implication:</strong> ${stratProf.q3_rr_implication}</div>
          <div style="font-size:0.84rem;color:var(--cyan)"><strong>4. Scouting Decision:</strong> ${stratProf.q4_scouting_decision}</div>
        </div>
      ` : ''}

      ${discBlock('Batting Profile', p.batting, true)}
      ${discBlock('Bowling Profile', p.bowling, false)}
    </div>
  `;

  document.getElementById('player-modal').classList.add('open');
}

function closeModal() {
  document.getElementById('player-modal')?.classList.remove('open');
}

// Initial Kick-off
renderTab();
