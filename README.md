# 🤖 RoboMaster Autonomous SLAM & Blaster Navigation System
> **Assignment 2 / Final Project**: ระบบสำรวจเขาวงกตอัตโนมัติ สร้างแผนที่ (SLAM), ตรวจจับเป้าหมายสีและรูปทรง และนำทางยิงกระสุนเจลด้วยอัลกอริทึม A* สำหรับหุ่นยนต์ **DJI RoboMaster EP**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![RoboMaster SDK](https://img.shields.io/badge/RoboMaster_SDK-EP-red.svg)](https://www.dji.com/robomaster-ep)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green.svg)](https://opencv.org/)
[![Tkinter](https://img.shields.io/badge/GUI-Tkinter_Dashboard-purple.svg)](https://docs.python.org/3/library/tkinter.html)

---

## 📌 สารบัญ (Table of Contents)
1. [ภาพรวมของระบบ (Overview)](#-ภาพรวมของระบบ-overview)
2. [สถาปัตยกรรมการทำงาน 2 รอบ (Dual-Round Architecture)](#-สถาปัตยกรรมการทำงาน-2-รอบ-dual-round-architecture)
   - [รอบที่ 1: การเดินสำรวจและสร้างแผนที่ (SLAM Mapping)](#-รอบที่-1-การเดินสำรวจและสร้างแผนที่-slam-mapping)
   - [รอบที่ 2: การนำทางและยิงเป้าหมาย (A* Navigation & Blasting)](#-รอบที่-2-การนำทางและยิงเป้าหมาย-a-navigation--blasting)
   - [โหมดรันต่อเนื่อง (Continuous Mode)](#-โหมดรันต่อเนื่อง-continuous-mode)
3. [ระบบจับเวลาภารกิจแต่ละรอบ (Mission Stopwatch System)](#-ระบบจับเวลาภารกิจแต่ละรอบ-mission-stopwatch-system)
4. [โครงสร้างโค้ดและโมดูล (Code Structure & Modules)](#-โครงสร้างโค้ดและโมดูล-code-structure--modules)
5. [อัลกอริทึมและการประมวลผล (Algorithms & Pipeline)](#-อัลกอริทึมและการประมวลผล-algorithms--pipeline)
   - [การตรวจจับกำแพงและการปรับกึ่งกลางช่อง (Wall Sensing & Recenter)](#การตรวจจับกำแพงและการปรับกึ่งกลางช่อง)
   - [การประมวลผลภาพและตรวจจับเป้าหมาย (Computer Vision)](#การประมวลผลภาพ-computer-vision)
   - [การวางแผนเส้นทาง (Path Planning: BFS & A*)](#การวางแผนเส้นทาง-path-planning)
6. [การติดตั้งและข้อกำหนดของระบบ (Installation & Requirements)](#-การติดตั้งและข้อกำหนดของระบบ-installation)
7. [วิธีใช้งานระบบ (Usage Guide)](#-วิธีใช้งานระบบ-usage-guide)
   - [การรันในโหมดจำลอง (Simulation Mode)](#1-โหมดจำลอง-simulation-mode---ไม่ใช้หุ่นยนต์จริง)
   - [การรันกับหุ่นยนต์จริง (Physical Robot Mode)](#2-โหมดหุ่นยนต์จริง-physical-robomaster-mode)
   - [คำสั่งผ่าน Terminal (Command Line Interface)](#3-ตัวเลือกผ่าน-command-line-cli-args)
8. [คู่มือหน้าจอ Dashboard (GUI Guide)](#-คู่มือหน้าจอ-dashboard-gui-guide)
9. [ผลลัพธ์และรายงานการทำงาน (Outputs & Evaluation)](#-ผลลัพธ์และรายงานการทำงาน-outputs--evaluation)
10. [ผู้พัฒนา (Credits)](#-ผู้พัฒนา-credits)

---

## 🌟 ภาพรวมของระบบ (Overview)

โปรเจกต์นี้เป็นการพัฒนาระบบควบคุมหุ่นยนต์ **DJI RoboMaster EP** เพื่อปฏิบัติภารกิจอัตโนมัติในพื้นที่เขาวงกตแบบกริด (ค่าเริ่มต้น 6×6 ช่อง ขนาดช่องละ 0.60 ม.) โดยมีจุดเด่นดังนี้:
- **ทำงานอัตโนมัติเต็มรูปแบบ (Fully Autonomous)** โดยไม่ต้องควบคุมด้วยมือ
- **ระบบสถาปัตยกรรมแบบ 2 รอบ (Dual-Round Architecture)**: รอบสำรวจทำแผนที่ (SLAM) และรอบนำทางความเร็วสูงเข้ายิงเป้าหมาย (A*)
- **ระบบปรับจูนกึ่งกลางช่องอัจฉริยะ (Mecanum Centering & IMU Alignment)** ป้องกันการดริฟต์สะสมของล้อ Mecanum
- **ระบบตรวจจับเป้าหมายด้วย Computer Vision (HSV & Contour Analysis)** จำแนก 4 สี (แดง, น้ำเงิน, เขียว, เหลือง) และ 4 รูปทรง (วงกลม, สี่เหลี่ยมจัตุรัส, แถบแนวนอน, แถบแนวตั้ง)
- **ระบบจับเวลาภารกิจแต่ละรอบ (Stopwatch)** เริ่มนับเวลาทันทีที่กดรัน บันทึกเวลาแยกรอบ 1, รอบ 2 และเวลารวม
- **หน้าจอ Dashboard สด (Tkinter GUI)** แสดงผลแผนที่ ตำแหน่งหุ่น ทิศทางกล้อง ภาพกล้องสด แกลเลอรีภาพถ่ายเป้าหมาย และระบบโหลดแผนที่ CSV

```mermaid
flowchart TD

subgraph group_control["Mission Control"]
  node_run["CLI Entry<br/>[run.py]"]
  node_coordinator["Mission Coordinator<br/>[slam.py]"]
  node_config["Runtime Configuration<br/>[config.py]"]
  node_navigation["Exploration and A*<br/>[navigation.py]"]
end

subgraph group_sensing["Sensing and Mapping"]
  node_vision["Target Vision<br/>[vision.py]"]
  node_state["Shared Mission State<br/>[state.py]"]
end

subgraph group_robot["Robot Operations"]
  node_robotcontrol["Robot Control<br/>[robot_control.py]"]
end

subgraph group_interface["Operator Interface"]
  node_dashboard["Live Dashboard<br/>[dashboard.py]"]
  node_preview["Vision Preview<br/>[vision_preview.py]"]
end

subgraph group_records["Mission Records"]
  node_map["Mission Map<br/>[mission_map.json]"]
  node_results["Logs and Snapshots"]
  node_evaluation["Evaluation and Reports<br/>[evaluation.py]"]
end

node_operator(("Operator"))
node_robot["RoboMaster EP / Simulator"]

node_operator -.->|"starts mission"| node_dashboard
node_operator -.->|"launches"| node_run
node_run -.->|"starts"| node_coordinator
node_dashboard -->|"starts mission"| node_coordinator
node_dashboard -->|"sets grid"| node_config
node_coordinator -->|"configures grid"| node_config
node_coordinator -.->|"plans routes"| node_navigation
node_coordinator -.->|"controls robot"| node_robotcontrol
node_robotcontrol -.->|"commands and reads"| node_robot
node_coordinator -.->|"processes targets"| node_vision
node_preview -->|"analyzes frames"| node_vision
node_preview -->|"connects camera"| node_robot
node_coordinator -->|"updates mission state"| node_state
node_coordinator -.->|"loads and saves"| node_map
node_coordinator -.->|"records mission"| node_results
node_coordinator -.->|"sends status"| node_dashboard
node_state -.->|"supplies live state"| node_dashboard
node_coordinator -.->|"requests reports"| node_evaluation
node_dashboard -.->|"imports"| node_evaluation

click node_run "https://github.com/praewa-japan1350/final-project/blob/main/run.py"
click node_coordinator "https://github.com/praewa-japan1350/final-project/blob/main/slam.py"
click node_config "https://github.com/praewa-japan1350/final-project/blob/main/config.py"
click node_navigation "https://github.com/praewa-japan1350/final-project/blob/main/navigation.py"
click node_dashboard "https://github.com/praewa-japan1350/final-project/blob/main/dashboard.py"
click node_preview "https://github.com/praewa-japan1350/final-project/blob/main/vision_preview.py"
click node_vision "https://github.com/praewa-japan1350/final-project/blob/main/vision.py"
click node_state "https://github.com/praewa-japan1350/final-project/blob/main/state.py"
click node_robotcontrol "https://github.com/praewa-japan1350/final-project/blob/main/robot_control.py"
click node_map "https://github.com/praewa-japan1350/final-project/blob/main/maps/mission_map.json"
click node_results "https://github.com/praewa-japan1350/final-project/tree/main/results"
click node_evaluation "https://github.com/praewa-japan1350/final-project/blob/main/evaluation.py"

classDef toneNeutral fill:#f8fafc,stroke:#334155,stroke-width:1.5px,color:#0f172a
classDef toneBlue fill:#dbeafe,stroke:#2563eb,stroke-width:1.5px,color:#172554
classDef toneAmber fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#78350f
classDef toneMint fill:#dcfce7,stroke:#16a34a,stroke-width:1.5px,color:#14532d
classDef toneRose fill:#ffe4e6,stroke:#e11d48,stroke-width:1.5px,color:#881337
classDef toneIndigo fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,color:#312e81
classDef toneTeal fill:#ccfbf1,stroke:#0f766e,stroke-width:1.5px,color:#134e4a
class node_run,node_coordinator,node_config,node_navigation toneBlue
class node_vision,node_state toneAmber
class node_robotcontrol toneMint
class node_dashboard,node_preview toneRose
class node_map,node_results,node_evaluation,node_operator,node_robot toneIndigo
