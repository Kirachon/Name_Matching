# Enhanced Name Matching Application - Implementation Plan & Ticket Tracker

**Version:** 1.0
**Last Updated:** YYYY-MM-DD
**Project Lead:** [Project Lead Name - Placeholder]

## 1. Project Overview
   - **Project Goal:** Develop a high-performance name matching application for Filipino identity data, enabling matching between two MySQL tables or a CSV input against a MySQL table. The solution leverages MySQL, Rust, GPU acceleration, and Podman for containerization.
   - **Key Objectives (from Dev Plan):**
     - **Accuracy**: Achieve >0.9 F1-score for Filipino names.
     - **Performance**: Process >10,000 record pairs/second (table-to-table) and <1min for CSV batches of 10,000 records.
     - **Scalability**: Support petabyte-scale datasets.
     - **Flexibility**: Handle same-format table matching and CSV-to-table matching.
     - **Security**: Ensure robust PII protection.
   - **Target Users:** [Specify target users, e.g., Data analysts, Government agencies]
   - **Key Stakeholders:** [Specify stakeholders, e.g., Project Sponsor, Development Team, End-User Representatives]

## 2. Scope (Summary from Dev Plan)
   - **In Scope:** Development of the name matching application including core logic, parsing, GPU acceleration, API, Podman containerization, UI for manual review, and features outlined in the three development phases of the "Enhanced Name Matching Application Development Plan with Podman".
   - **Out of Scope:** (Refer to detailed dev plan for specifics if any were mentioned; otherwise, state "To be defined if specific exclusions arise.")

## 3. Phases & Milestones (Derived from Dev Plan)

   - **Phase 1: Core CPU Matching Engine & Data Foundation (Months 1-6)**
     - **Goal:** Build a functional baseline with robust parsing, CPU-based matching, and flexible input handling.
     - **Estimated Duration:** 6 Months
     - **Key Deliverables (from Dev Plan):** MySQL schema and indexes, Rust service for CSV import and table matching, Initial Filipino name dictionary and preprocessing routines, Podman container definitions for Rust and MySQL, Benchmark report (Accuracy >0.8 F1-score, Throughput >1,000 pairs/second CPU).

   - **Phase 2: GPU Acceleration and Performance Optimization (Months 7-12)**
     - **Goal:** Accelerate matching with GPU kernels and optimize for large-scale table comparisons.
     - **Estimated Duration:** 6 Months
     - **Key Deliverables (from Dev Plan):** GPU-accelerated matching pipeline, Optimized blocking strategies, Security measures for PII handling, Updated Podman containers with GPU support, Benchmark report (>10,000 pairs/second GPU, <1min for 10,000-record CSV).

   - **Phase 3: Advanced Features, Scalability, and Production Hardening (Months 13-18)**
     - **Goal:** Add advanced features, ensure scalability, and prepare for production.
     - **Estimated Duration:** 6 Months
     - **Key Deliverables (from Dev Plan):** Production-ready application with UI & XAI, Extended API, Scalable architecture for petabyte-scale data, Comprehensive security and resilience measures, Podman-based Kubernetes deployment scripts, Final benchmark report (all KPIs met).

## 4. Task & Ticket Tracker

### Phase 1: Core CPU Matching Engine & Data Foundation Tickets (Months 1-6)
| Ticket ID | Task Description                                       | Assignee      | Priority | Status      | Due Date   | Notes/Blockers |
|-----------|--------------------------------------------------------|---------------|----------|-------------|------------|----------------|
| P1-001    | Finalize MySQL Schema Design (hh_id, names, geo, etc.) | [Dev Team]    | High     | To Do       | YYYY-MM-DD | Schema details in dev plan; utf8mb4_unicode_ci; B-Tree/FULLTEXT indexes |
| P1-002    | Implement CSV Input Handling (parsing, validation, temp table creation) | [Dev Team]    | High     | To Do       | YYYY-MM-DD | Use `csv` crate; TOML/YAML config for mapping; batch inserts |
| P1-003    | Develop Rust Orchestration Service (table/CSV inputs, sqlx for async MySQL) | [Dev Team]    | High     | To Do       | YYYY-MM-DD | Connection pooling; candidate generation query construction |
| P1-004    | Build Parsing and Preprocessing Logic (middle_name_last_name, Unicode normalization, date/geo standardization) | [Dev Team]    | High     | To Do       | YYYY-MM-DD | Filipino patterns dictionary; NFKC; `chrono`; PSGC gazetteer |
| P1-005    | Implement CPU-Based Matching Algorithms (Jaro-Winkler, Soundex, Jaccard Index) | [Dev Team]    | High     | To Do       | YYYY-MM-DD | Use `fuzzy_name_match` crate; apply to relevant fields |
| P1-006    | Develop Scoring and Decisioning Model (weighted scores, thresholds for Match/Non-Match/Manual Review) | [Dev Team]    | Medium   | To Do       | YYYY-MM-DD | e.g., 40% names, 30% birthdate, 30% geography |
| P1-007    | Create Testing Framework & Curate Filipino Name Test Dataset | [QA Team]     | Medium   | To Do       | YYYY-MM-DD | Verified matches/non-matches; benchmark accuracy/throughput |
| P1-008    | Implement Logging (tracing) and Monitoring (prometheus) Setup | [Dev Team]    | Medium   | To Do       | YYYY-MM-DD | Structured logging for CSV import, query performance; track metrics |
| P1-009    | Setup Podman for Rust App & MySQL (dev/test containers, rootless execution) | [DevOps/Team] | High     | To Do       | YYYY-MM-DD | Docker-compatible CLI for images with CUDA/OpenCL dependencies |

### Phase 2: GPU Acceleration and Performance Optimization Tickets (Months 7-12)
| Ticket ID | Task Description                                       | Assignee      | Priority | Status      | Due Date   | Notes/Blockers |
|-----------|--------------------------------------------------------|---------------|----------|-------------|------------|----------------|
| P2-001    | Profile Phase 1 Application for Bottlenecks (`cargo flamegraph`) | [Dev Team]    | High     | To Do       | YYYY-MM-DD | Identify Jaro-Winkler bottlenecks etc. |
| P2-002    | Develop GPU Kernels (CUDA/OpenCL for Jaro-Winkler, Levenshtein, Jaccard on n-grams) | [GPU Dev]     | High     | To Do       | YYYY-MM-DD | Use `cudarc` or `opencl3`; shared memory optimization |
| P2-003    | Implement Data Marshalling for GPU (batching by length, pinned host memory, CUDA streams) | [GPU Dev]     | High     | To Do       | YYYY-MM-DD | Asynchronous transfers |
| P2-004    | Integrate GPU Kernels into Rust Service via FFI (coarse & fine pass pipeline) | [Dev Team]    | High     | To Do       | YYYY-MM-DD | Efficient data serialization; filter with coarse pass |
| P2-005    | Optimize MySQL Blocking (phonetic keys - Soundex; prototype GPU-accelerated blocking) | [Dev Team]    | Medium   | To Do       | YYYY-MM-DD | e.g., n-gram profile computation |
| P2-006    | Perform GPU Performance Tuning (Nsight Compute, MySQL EXPLAIN) | [GPU Dev/DBA] | High     | To Do       | YYYY-MM-DD | Tune kernel block sizes; target >10k pairs/sec |
| P2-007    | Implement Security Enhancements (TLS for MySQL/API, MySQL TDE, sanitize inputs, zero GPU memory) | [Dev Team/Sec] | High     | To Do       | YYYY-MM-DD |  |
| P2-008    | Update Podman Containers for GPU Support & Test Access (`--gpus` flag, K8s compatibility) | [DevOps/Team] | High     | To Do       | YYYY-MM-DD | NVIDIA CUDA drivers, OpenCL runtimes |

### Phase 3: Advanced Features, Scalability, and Production Hardening Tickets (Months 13-18)
| Ticket ID | Task Description                                       | Assignee      | Priority | Status      | Due Date   | Notes/Blockers |
|-----------|--------------------------------------------------------|---------------|----------|-------------|------------|----------------|
| P3-001    | Build Manual Review UI (React) with XAI features (scores, rule activations, SHAP) | [Frontend/Team] | High     | To Do       | YYYY-MM-DD | Integrate with API for review tasks |
| P3-002    | Extend API Endpoints (`/match/table`, `/match/batch/{job_id}`, config, HATEOAS) | [Backend Team] | High     | To Do       | YYYY-MM-DD |  |
| P3-003    | Implement Advanced Blocking (Sorted Neighborhood Method or LSH in Rust; scale GPU blocking) | [Dev Team]    | Medium   | To Do       | YYYY-MM-DD | For petabyte datasets |
| P3-004    | Prototype Deep Learning Similarity Scoring (Transformer/Sentence-BERT via ONNX Runtime) | [R&D/Team]    | Medium   | To Do       | YYYY-MM-DD | Evaluate accuracy vs. traditional |
| P3-005    | Implement MySQL Sharding & K8s Stateless Deployment (Podman for pod specs) | [DevOps/DBA]  | High     | To Do       | YYYY-MM-DD |  |
| P3-006    | Implement Resilience Features (retries, circuit breakers, DLQs; Prometheus/Grafana monitoring) | [Dev Team]    | High     | To Do       | YYYY-MM-DD | Alerts for anomalies |
| P3-007    | Enforce RBAC for MySQL/Rust/UI & Explore PPRL Techniques (Bloom filters) | [Security Team] | Medium   | To Do       | YYYY-MM-DD |  |
| P3-008    | Finalize Podman Containers & Kubernetes Deployment Scripts (GPU allocation, external config) | [DevOps/Team] | High     | To Do       | YYYY-MM-DD | TOML/YAML config, secrets via env vars |
| P3-009    | Conduct Final Production-like Testing & Benchmarking (all KPIs) | [QA/Team]     | High     | To Do       | YYYY-MM-DD |  |

## 5. Dependencies & Risks
   - **Dependency:** Availability of GPU resources for development and testing.
   - **Dependency:** Expertise in Rust, CUDA/OpenCL, and MySQL optimization.
   - **Risk:** Complexity of parsing Filipino `middle_name_last_name` field leading to lower accuracy. (Mitigation: Iterative refinement, dictionary expansion, rule-based and statistical approaches).
   - **Risk:** Achieving performance targets for GPU kernels. (Mitigation: Detailed profiling with Nsight Compute, expert consultation, iterative optimization).
   - **Risk:** Data security breaches, especially with PII and CSV handling. (Mitigation: Strict adherence to security best practices like TLS, TDE, input sanitization, zeroing memory, regular audits).
   - **Risk:** Scalability challenges with petabyte-scale data. (Mitigation: MySQL sharding, stateless K8s services, efficient blocking).
   - *(Add more as identified during project lifecycle)*

## 6. Communication Plan
   - **Regular Meetings:** 
     - Daily Stand-ups (Development Team)
     - Weekly Progress Reviews (Project Lead, Team Leads)
     - Bi-Weekly Stakeholder Updates (Project Lead, Key Stakeholders)
   - **Reporting:** 
     - Weekly internal progress summaries.
     - Bi-weekly formal status reports to stakeholders.
   - **Tools:** 
     - This Markdown document for overall plan and high-level ticket tracking.
     - [Specify Issue Tracker e.g., Jira, Trello, GitHub Issues] for detailed task management if adopted.
     - [Specify Communication Platform e.g., Slack, Microsoft Teams] for daily communication.
     - [Specify Document Repository e.g., Confluence, SharePoint, Git Repository] for all project documentation.

## 7. Sign-off
   - **Project Lead:** _________________________ Date: __________
   - **Key Stakeholder (Example):** _________________________ Date: __________
   - *(Add more stakeholder sign-off lines as needed)*