"""Premium browser UI shell for the AtomicVision Hugging Face Space."""

from __future__ import annotations

from textwrap import dedent


def render_home_html() -> str:
    """Return the premium landing page + live demo UI for the Space root route."""

    return dedent(
        """\
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <meta
              name="description"
              content="Luxury AI platform for non-destructive multi-defect characterization using PyTorch."
            >
            <meta name="theme-color" content="#050505">
            <meta property="og:title" content="AtomicVision | AI Defect Mapping Platform">
            <meta
              property="og:description"
              content="Luxury AI platform for non-destructive multi-defect characterization using PyTorch."
            >
            <meta property="og:type" content="website">
            <title>AtomicVision | AI Defect Mapping Platform</title>
            <link rel="stylesheet" href="/static/space-ui.css">
          </head>
          <body>
            <div class="cursor-ring" aria-hidden="true"></div>
            <div class="particles" aria-hidden="true"></div>

            <header class="site-nav">
              <a class="brand-mark" href="#hero" aria-label="AtomicVision home">
                <span class="brand-orbit" aria-hidden="true"></span>
                <span>ATOMICVISION</span>
              </a>
              <nav aria-label="Primary navigation">
                <a href="#problem">Problem</a>
                <a href="#solution">System</a>
                <a href="#demo">Demo</a>
                <a href="#results">Results</a>
                <a href="#team">Team</a>
              </nav>
            </header>

            <main>
              <section id="hero" class="hero-scroll">
                <div class="hero-sticky">
                  <div class="hero-grid">
                    <div class="hero-copy" data-reveal>
                      <div class="eyebrow">Metaminds / Deep Materials AI</div>
                      <h1>ATOMICVISION</h1>
                      <p>Autonomous AI for Non-Destructive Multi-Defect Mapping</p>
                      <div class="hero-tags" aria-label="Technology tags">
                        <span>OpenEnv runtime</span>
                        <span>DefectNet-lite prior</span>
                        <span>Hugging Face Space</span>
                      </div>
                      <div class="hero-actions">
                        <a class="btn btn-primary" href="#demo">Launch Demo <span aria-hidden="true">&rarr;</span></a>
                        <a class="btn btn-secondary" href="https://github.com/Adityabaskati-weeb/AtomicVision" target="_blank" rel="noreferrer">GitHub <span aria-hidden="true">&nearr;</span></a>
                        <a class="btn btn-ghost" href="https://github.com/Adityabaskati-weeb/AtomicVision/blob/main/blog.md" target="_blank" rel="noreferrer">Research <span aria-hidden="true">&rsaquo;</span></a>
                      </div>
                    </div>

                    <div class="hero-visual" data-reveal>
                      <div class="video-frame">
                        <video
                          src="/static/media/Atoms_Move_Slowly_Around_Circle.mp4"
                          muted
                          playsinline
                          preload="auto"
                          aria-label="Atomic orbital transition visualization"
                        ></video>
                        <div class="video-gradient"></div>
                        <div class="video-hud">
                          <span id="hud-mode">Scroll scrub / live orbit</span>
                          <span id="hud-status">OpenEnv ready</span>
                        </div>
                      </div>
                      <div class="hero-data-card glass-card">
                        <span class="panel-kicker">Live Space Runtime</span>
                        <h3>Investor-grade interface, real OpenEnv backend.</h3>
                        <ul class="detail-list">
                          <li><span>Runtime</span><strong id="hero-runtime">Docker Space</strong></li>
                          <li><span>Deployment</span><strong>AtomicVision OpenEnv</strong></li>
                          <li><span>Model</span><strong id="hero-model">Hard recall micro boost</strong></li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              </section>

              <section id="problem" class="section section-tight">
                <div class="section-heading" data-reveal>
                  <span class="eyebrow">Problem</span>
                  <h2>Destructive testing breaks the signal before intelligence can scale.</h2>
                </div>
                <div class="card-grid three">
                  <article class="glass-card problem-card" data-reveal>
                    <span class="line-icon microscope" aria-hidden="true"></span>
                    <h3>Slow destructive sampling</h3>
                    <p>Legacy inspection cuts into specimens, adds lab delays, and loses spatial context across multi-defect systems.</p>
                  </article>
                  <article class="glass-card problem-card" data-reveal>
                    <span class="line-icon radar" aria-hidden="true"></span>
                    <h3>Sparse defect visibility</h3>
                    <p>Spectral signatures hide overlapping fracture, void, oxide, and strain modes inside noisy measurement windows.</p>
                  </article>
                  <article class="glass-card problem-card" data-reveal>
                    <span class="line-icon waves" aria-hidden="true"></span>
                    <h3>Unscalable interpretation</h3>
                    <p>Human review struggles to convert high-volume spectra into repeatable, investor-grade operating intelligence.</p>
                  </article>
                </div>
              </section>

              <section id="solution" class="section">
                <div class="section-heading narrow" data-reveal>
                  <span class="eyebrow">Solution</span>
                  <h2>A monochrome AI pipeline for spectral defect intelligence.</h2>
                </div>
                <div class="glass-card pipeline-card" data-reveal>
                  <div class="pipeline-node">
                    <div class="pipeline-icon"><span class="line-icon scan" aria-hidden="true"></span></div>
                    <span>Spectrum Input</span>
                    <div class="pipeline-line"></div>
                  </div>
                  <div class="pipeline-node">
                    <div class="pipeline-icon"><span class="line-icon chip" aria-hidden="true"></span></div>
                    <span>DefectNet</span>
                    <div class="pipeline-line"></div>
                  </div>
                  <div class="pipeline-node">
                    <div class="pipeline-icon"><span class="line-icon brain" aria-hidden="true"></span></div>
                    <span>AI Attention</span>
                    <div class="pipeline-line"></div>
                  </div>
                  <div class="pipeline-node">
                    <div class="pipeline-icon"><span class="line-icon layers" aria-hidden="true"></span></div>
                    <span>Defect Output</span>
                  </div>
                </div>
              </section>

              <section id="demo" class="section demo-section">
                <div class="section-heading" data-reveal>
                  <span class="eyebrow">Demo</span>
                  <h2>Run a live OpenEnv-backed defect analysis in one polished flow.</h2>
                </div>
                <div class="demo-grid">
                  <div class="glass-card upload-panel" data-reveal>
                    <div class="panel-row">
                      <span class="panel-kicker">Sample ingest</span>
                      <div class="difficulty-switch" aria-label="Scenario difficulty">
                        <button class="difficulty-pill active" type="button" data-difficulty="medium">Medium</button>
                        <button class="difficulty-pill" type="button" data-difficulty="hard">Hard</button>
                      </div>
                    </div>
                    <label class="upload-zone">
                      <input id="sample-file" type="file" accept=".csv,.txt,.json,text/csv,application/json,text/plain">
                      <span class="line-icon upload" aria-hidden="true"></span>
                      <span id="file-label">CSV, TXT, or JSON spectral file</span>
                    </label>
                    <button id="analyze-button" class="btn btn-primary analyze-button">Analyze Sample <span aria-hidden="true">&rarr;</span></button>
                    <p class="demo-note" id="demo-note">Upload a spectrum to drive live backend predictions, or run the built-in OpenEnv episode path with no file attached.</p>
                  </div>
                  <div class="glass-card model-panel" data-reveal>
                    <span class="panel-kicker">Runtime capsule</span>
                    <h3>PyTorch spectral reasoning with a premium operating surface.</h3>
                    <p>Behind the cinematic shell, the demo talks to the actual OpenEnv endpoints that power the Hugging Face Space.</p>
                    <ul class="detail-list runtime-list">
                      <li><span>Episode</span><strong id="runtime-episode">Awaiting analysis</strong></li>
                      <li><span>Material ID</span><strong id="runtime-material">-</strong></li>
                      <li><span>Host family</span><strong id="runtime-family">-</strong></li>
                      <li><span>Budget remaining</span><strong id="runtime-budget">-</strong></li>
                      <li><span>Environment note</span><strong id="runtime-message">OpenEnv lab online</strong></li>
                    </ul>
                  </div>
                </div>
              </section>

              <section id="results" class="section">
                <div class="section-heading narrow" data-reveal>
                  <span class="eyebrow">Results Dashboard</span>
                  <h2>High-confidence defect outputs in an operator-grade panel.</h2>
                </div>
                <div class="glass-card dashboard" data-reveal>
                  <div class="metrics-row">
                    <div class="metric"><span>Signal Fidelity</span><strong id="metric-fidelity">--</strong></div>
                    <div class="metric"><span>Inference Latency</span><strong id="metric-latency">--</strong></div>
                    <div class="metric"><span>Map Resolution</span><strong id="metric-resolution">--</strong></div>
                    <div class="metric"><span>Model Certainty</span><strong id="metric-certainty">--</strong></div>
                  </div>
                  <div class="dashboard-summary" id="results-summary">Upload a sample and launch the demo to populate live defect predictions.</div>
                  <div class="defect-list" id="defect-list">
                    <div class="defect-row placeholder-row">
                      <div><strong>Awaiting analysis</strong><span>Live OpenEnv results will appear here.</span></div>
                      <div class="confidence"><div><span style="width: 0%"></span></div><b>--</b></div>
                    </div>
                  </div>
                </div>
              </section>

              <section id="explainability" class="section explain-section">
                <div class="section-heading" data-reveal>
                  <span class="eyebrow">Explainability</span>
                  <h2>Attention maps reveal where the model sees structural risk.</h2>
                </div>
                <div class="explain-grid">
                  <div class="glass-card heatmap-card" data-reveal>
                    <h3>Attention Heatmap</h3>
                    <div class="heatmap" id="heatmap" aria-label="Attention heatmap"></div>
                  </div>
                  <div class="glass-card graph-card" data-reveal>
                    <h3>Spectral Relevance</h3>
                    <div class="bar-graph" id="bar-graph" aria-label="Spectral relevance graph"></div>
                  </div>
                </div>
              </section>

              <section id="team" class="section team-section">
                <div class="glass-card team-card" data-reveal>
                  <span class="eyebrow">Team</span>
                  <h2>Metaminds</h2>
                  <div class="team-points">
                    <span>Autonomous materials intelligence</span>
                    <span>Spectral foundation modeling</span>
                    <span>Non-destructive defect characterization</span>
                  </div>
                </div>
              </section>
            </main>

            <footer class="site-footer">
              <div>
                <strong>ATOMICVISION</strong>
                <span>Metaminds advanced materials AI platform</span>
              </div>
              <nav aria-label="Footer links">
                <a href="https://github.com/Adityabaskati-weeb/AtomicVision" target="_blank" rel="noreferrer">GitHub</a>
                <a href="https://huggingface.co/spaces/prodigyhuh/atomicvision-openenv" target="_blank" rel="noreferrer">Hugging Face</a>
                <a href="https://github.com/Adityabaskati-weeb/AtomicVision/blob/main/blog.md" target="_blank" rel="noreferrer">Research</a>
              </nav>
            </footer>

            <script>
              window.ATOMICVISION_CONFIG = {
                resetUrl: "/reset",
                stepUrl: "/step",
                stateUrl: "/state",
                wsUrl: "/ws",
                analyzeUploadUrl: "/analyze_upload",
              };
            </script>
            <script src="/static/space-ui.js" defer></script>
          </body>
        </html>
        """
    ).strip()
