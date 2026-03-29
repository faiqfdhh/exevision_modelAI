

CAT405 System Requirements and Design Report Academic Session: 2025/2026

i


School of Computer Sciences
CAT405 Intelligent Computing Major Project
System Requirements and Design Report
ExeVision
An AI-Powered Camera System for Evaluation of Exercises

## MUHAMMAD FAIQ FADHLULLAH
## 160677

## Supervisor: Dr. Azman Bin Abdul Malik
## Examiner: Dr. Fadratul Hafinaz Hassan

## Academic Session
## 2025/2026

CAT405 System Requirements and Design Report Academic Session: 2025/2026

ii
## DECLARATION

“I  declare  that  the  following  is  my  own  work  and  does  not  contain  any
unacknowledged work from any other sources.  This report was undertaken to fulfill
the  requirements  of  the  Undergraduate  Major  Project  for  the  Bachelor  of  Science  in
Computer Science (Honors) program at Universiti Sains Malaysia”.


## Signature:
## Name: Muhammad Faiq Fadhlullah
## Date 15
th
## December 2025











CAT405 System Requirements and Design Report Academic Session: 2025/2026

iii
## ABSTRAK
Umum  mengetahui  senaman  mampu  memberi  pelbagai  kebaikan  kepada  kesihatan,
baik  dari  segi  fizikal  mahupun  mental.Namun,  masih  ramai  yang  bukan  sahaja  tidak
bersenam,  tapi  dalam  kalangan  orang  yang  aktif  sekalipun,  tahap  penguasaan    cara
melakukan  senaman  dengan  betul  kekal  rendah.  Oleh  itu,  projek  ExeVision  ini  akan
menghasilkan   suatu   aplikasi   yang   mampu   menilai,   memberi   maklumbalas   dan
sekaligus membimbing pengguna untuk melakukan senaman dengan betul dan  selamat.
Penilaian ini   akan   dilaksanakan   melalui   teknologi   Penglihatan   Komputer   yang
mengetengahkan teknik Hibrid antara Peraturan Simbolik  sebagai penilai utama  dan
Teknik Neural seperti  Bi-LSTM sebagai  penilai tambahan.  Projek  ExeVision  ini
menawarkan dua saluran penggunaan utama, iaitu Edge Mode yang menggunakan edge
computing melalui  penggunaan kamera Luxonis OAK-D dan Nvidia Jetson Orin Nano
sebagai  unit  pemprosesan,  dan  juga Cloud  Mode  melalui  teknologi  tanpa  pelayan.
Projek ini menandakan kemajuan dalam teknologi AQA (Action Quality Assessment)
yang  rata-rata  hanya  memberi  penilaian  bersifat  binari  dan    klasifikasi  sahaja.
Sehubungan  dengan  itu,  projek  ini  juga  menyokong  SDG  (Sustainable  Development
Goal) 3 dengan mempromosi aktiviti fizikal yang lebih selamat, efektif dan berkesan.
Kata  Kunci: Senaman,  Penglihatan  Komputer,  teknik  hibrid,    edge  computing,  ,
teknologi tanpa pelayan, AQA, , SDG 3



CAT405 System Requirements and Design Report Academic Session: 2025/2026

iv
## ABSTRACT
It  is  generally  known  that  exercise  can  provide  various  benefits  to  health,  both
physically  and  mentally.  However,  many  people  not  only  do  not  exercise,  but  even
among  those  who  are  active,  the  level  of  mastery  in  performing  exercises  correctly
remains low. Therefore, this ExeVision project will produce an application capable of
evaluating, providing feedback, and simultaneously guiding users to perform exercises
correctly  and  safely.  This  evaluation  will  be  implemented  through  Computer  Vision
technology that highlights a Hybrid technique between Symbolic Rules as the primary
evalutor and  Neural  methods,  such  as  Bi-LSTM,  as  the supplementar evaluator.  The
ExeVision  project  offers  two  main  usage  channels:  Edge  Mode,  which  utilizes  edge
computing through the use of a Luxonis OAK-D camera and Nvidia Jetson Orin Nano
as the processing unit, and Cloud Mode via serverless technology. This project marks
an  advancement  in  Action  Quality  Assessment  (AQA)  technology,  which  typically
provides  only  binary  and  classification-based  evaluations.  In  this  regard,  this  project
also  supports  SDG  (Sustainable  Development  Goal)  3  by  promoting  safer,  more
effective, and impactful physical activity.
Keywords: Exercise,  Computer  Vision,  hybrid  technique,  edge  computing,  serverless
technology, AQA, SDG 3


CAT405 System Requirements and Design Report Academic Session: 2025/2026

v
## ACKNOWLEDGEMENTS
I would like to express my sincere gratitude to my supervisor, Dr. Azman Bin Abdul
Malik, for all of his help, encouragement, and constructive criticism while working on
this  project.  His  knowledge  and  support  have  been  crucial  in  determining  the  course
and outcome of this project. I would like to thank my final year project examiner,  Dr.
Fadratul Hafinaz Hassan, for serving as the evaluator and investing the time and effort
to study my project and report. I am also deeply thankful to Dr. Chew Xin Ying for her
role  as  the  final project coordinator,  providing  essential  resources  that  significantly
contributed to the development and execution of this project.
I  am  incredibly  grateful  to  my  family  for  their  unwavering  support,  love,  and
understanding during this journey. Their steadfast assistance has been the foundation
of my strength and perseverance. I wish to also express my gratitude to my friends for
their support, empathy, and inspiration, all of which have been  crucial in  helping me
stay balanced and optimistic throughout this project.


CAT405 System Requirements and Design Report Academic Session: 2025/2026

vi
## TABLE OF CONTENTS
DECLARATION ........................................................................................................... ii
ABSTRAK .................................................................................................................... iii
ABSTRACT .................................................................................................................. iv
ACKNOWLEDGEMENTS ........................................................................................... v
TABLE OF CONTENTS .............................................................................................. vi
LIST OF FIGURES ...................................................................................................... ix
LIST OF TABLES ......................................................................................................... x
LIST OF ABBREVIATIONS AND SYMBOLS ......................................................... xi
1 INTRODUCTION ................................................................................................. 1
1.1 Project Background ........................................................................................ 1
1.2 Problem Statements ....................................................................................... 3
1.3 Motivation ...................................................................................................... 5
1.4 System Objectives .......................................................................................... 6
1.5 Proposed Solutions ......................................................................................... 6
1.6 Project Module ............................................................................................... 8
1.7 Benefits / Impact / Significance of Project .................................................. 10
1.8 Uniqueness of Proposed Solutions ............................................................... 11
1.9 Organization of Report ................................................................................ 12
2 BACKGROUND AND RELATED WORK ....................................................... 13
2.1 Status of the Project ..................................................................................... 13

CAT405 System Requirements and Design Report Academic Session: 2025/2026

vii
2.2 Project Context ............................................................................................. 13
2.3 Comparison Analysis of Similar Projects .................................................... 14
2.4 Related Projects’s Strength and Weaknesses ............................................... 16
2.5 Brief Introduction of Proposed Solution ...................................................... 19
3 SYSTEM REQUIREMENTS AND ANALYSIS ............................................... 20
3.1 Project Scope , Capabilities and Limitations ............................................... 20
3.1.1 Project Deliverables ............................................................................. 20
3.1.2 Project Exclusions ................................................................................ 20
3.1.3 Project Stakeholders ............................................................................. 20
3.1.4 Project Capabilities .............................................................................. 20
3.1.5 Project Limitations ............................................................................... 21
Project Management ................................................................................................ 22
3.1.6 Work Breakdown Structure ................................................................. 22
3.1.7 Gantt Chart ........................................................................................... 23
3.1.8 Milestone Timeline .............................................................................. 24
3.2 Development Methodology ......................................................................... 25
3.3 SWOT Analysis ........................................................................................... 26
4 SYSTEM DESIGN AND IMPLEMENTATION ................................................ 27
4.1 Diagrams ...................................................................................................... 27
4.1.1 Use Case Diagram ................................................................................ 27
4.1.2 Use Case Diagram Description ............................................................ 28

CAT405 System Requirements and Design Report Academic Session: 2025/2026

viii
4.1.3 Sequence Diagram ............................................................................... 36
4.1.4 Entity Relationship Diagram ................................................................ 41
4.2 Detailed Description of Project .................................................................... 42
4.2.1 Functional Requirements ..................................................................... 42
4.2.2 Non-Functional Requirements ............................................................. 44
4.2.3 Flowcharts of AI Module ..................................................................... 46
4.2.4 Architecture Diagram ........................................................................... 47
4.3 Intelligent Methods Used ............................................................................. 48
4.3.1 Computer Vision & Feature Extraction ............................................... 48
4.3.2 State-Aware Temporal Segmentation .................................................. 48
4.3.3 Evaluation & Reasoning  Model .......................................................... 48
4.3.4 Decision Feedback Generation ............................................................ 49
4.4 Data Sources ................................................................................................ 49
4.5 Technology Deployed .................................................................................. 50
4.5.1 Hardware .............................................................................................. 50
4.5.2 Software ............................................................................................... 51
5 CONCLUSION .................................................................................................... 52
6 SDG ALIGNMENT ............................................................................................. 53
7 REFERENCES .................................................................................................... 54
8 APPENDIX .......................................................................................................... 58



CAT405 System Requirements and Design Report Academic Session: 2025/2026

ix
## LIST OF FIGURES
Figure 1: Flow of Cloud Mode ...................................................................................... 7
Figure 2: Flow of Edge Mode ........................................................................................ 7
Figure 3: Project Modules .............................................................................................. 8
Figure 4: Generic AQA Pipeline .................................................................................. 13
Figure 5: ExeVision Cloud Mode Use Case Diagram ................................................. 27
Figure 6: ExeVision Edge Mode Use Case Diagram ................................................... 27
Figure 7: Create Account Sequence Diagram .............................................................. 36
Figure 8: Log In Sequence Diagram ............................................................................ 37
Figure 9: Forgot Password Sequence Diagram ............................................................ 38
Figure 10: Cloud Mode Sequence Diagram ................................................................. 39
Figure 11: Project ERD ................................................................................................ 41
Figure 12: AI Module Flowchart ................................................................................. 46
Figure 13: Project Architecture Diagram ..................................................................... 47










CAT405 System Requirements and Design Report Academic Session: 2025/2026

x
## LIST OF TABLES
Table 1: Comparison of Current Solutions .................................................................. 18
Table 2: UC-001 Description ....................................................................................... 28
Table 3:UC-002 Description ........................................................................................ 29
Table 4: UC-003 Description ....................................................................................... 30
Table 5:UC-004 Description ........................................................................................ 31
Table 6: UC-005 Description ....................................................................................... 32
Table 7:UC-006 Description ........................................................................................ 33
Table 8: UC-07 Description ......................................................................................... 34
Table 9:UC-008 Description ........................................................................................ 35
Table 10: Functional Requirements ............................................................................. 43
Table 11: Non-Functional Requirements ..................................................................... 45
Table 12: NVIDIA Jetson Orin Nano Specifications .................................................. 50
Table 13: Luxonis OAK-D Specification .................................................................... 50
Table 14: HP Victus 15 Specifications ........................................................................ 51
Table 15: Software Stack ............................................................................................. 51







CAT405 System Requirements and Design Report Academic Session: 2025/2026

xi
## LIST OF ABBREVIATIONS AND SYMBOLS
AES - Advanced Encryption Standard
AI - Artificial Intelligence
API - Application Programming Interface
AQA - Action Quality Assessment
B2B - Business-to-Business
B2C - Business-to-Consumer
BiLSTM - Bidirectional Long Short-Term Memory
CPU - Central Processing Unit
CUDA - Compute Unified Device Architecture
DDR4 - Double Data Rate 4
DTW - Dynamic Time Warping
ERD - Entity Relationship Diagram
FPS - Frames Per Second
Gbps - Gigabits per second
GDPR - General Data Protection Regulation
GPU - Graphics Processing Unit
GUI - Graphical User Interface
IoT - Internet of Things
LLM - Large Language Model

CAT405 System Requirements and Design Report Academic Session: 2025/2026

xii
LPDDR5 - Low-Power Double Data Rate 5
MB - Megabyte
MHz - Megahertz
ML - Machine Learning
MOV - QuickTime File Format
MP - Megapixel
MP4 - MPEG-4 Part 14
NCDs - Non-Communicable Diseases
PT - Personal Trainer
RF - Random Forest
RGB-D - Red Green Blue - Depth
SDG - Sustainable Development Goal
SDK - Software Development Kit
SEMG - Surface Electromyography
SMPLX - Skinned Multi-Person Linear Model (Extended)
SSD - Solid State Drive
SVR - Support Vector Regressors
USB - Universal Serial Bus
VPU - Vision Processing Unit
WHO - World Health Organization

CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 1
## 1 INTRODUCTION
## 1.1 Project Background
Regular exercise is widely recognized as essential for physical and mental health, yet
in Malaysia, few people actually perform any physical exercise more than brisk walking
or jogging. Particularly, the fitness culture in Malaysia is not widespread and remains
sparse. Even  within the fitness community  itself,  early quitters  are common.  Studies
report that between 40 and 65 percent of new members discontinue within the first three
to six months of joining a fitness centre (Gjestvang et al., 2019; Gjestvang et al., 2023),
while  intervention  programs  show  that  roughly  half  of  participants  stop  within  six
months even under structured supervision (Collins et al., 2022). These figures indicate
that the early stages of training are the most fragile and that many individuals struggle
to consistently perform their exercises.
A frequently cited factor contributing to exercise discontinuation is the perceived lack
of progress or improvement in physical performance and body composition during the
early  stages  of  training.  When  individuals  fail  to  observe  tangible  outcomes,  self-
efficacy  and  intrinsic  motivation  often  decline,  leading  to  reduced  participation  and
eventual  dropout  (Ingledew and Markland,  2008;  Ntoumanis  et  al.,  2021).  This
perception is particularly prevalent among novice exercisers, who may lack adequate
understanding  of  training  principles,  progression,  and  proper  exercise  execution
(Teixeira et al., 2012). Empirical evidence suggests that improper movement patterns
or suboptimal training intensity can significantly hinder physiological adaptations and
visible progress (Gentil et al., 2017; Schoenfeld, 2010). Consequently, the inability to
perform  exercises  with  correct  form  and  intensity  not  only  limits  results  but  also
reinforces  negative beliefs about  one’s capability to  improve, thereby exacerbating
early dropout rates.
Research  in  biomechanics  shows  that  real-time  feedback  can  significantly  improve
movement  technique.  Visual  or  verbal  cues  provided  during  exercise  enhance  joint
control and landing mechanics, improving both safety and learning outcomes (Nyman
and Armstrong, 2015; Neilson et al., 2019; Storberget et al., 2017; Agresta and Brown,
2015). Professional trainers are the most common avenue of such corrective feedback,

CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 2
yet  their  services  are  often  costly  or  inaccessible  for  casual  gym  users.  Public  health
surveys  consistently  report  financial  and  knowledge  barriers  as  key  reasons  for
inactivity  (CDC,  2025;  Koh  et  al.,  2022).  These  findings  reveal  a  clear  need  for  an
accessible and intelligent system that helps individuals monitor and correct their form
in  real  time.  By  offering  immediate,  data-driven  feedback  without  the  cost  or
dependency  on  personal  trainers,  such  technology  can  bridge  the  gap  between
professional guidance and independent training.
There are several existing solutions that addresses this. Train Fitness, a fitness company
focusing  on  automated  exercise  tracking  is  one  of  them.  They  rely  on  wrist-worn
wearables, such as the Apple Watch to analyse complex movement patterns in real-time
through a proprietary algorithm called Neural Kinetic Profiling. However, the reliance
of  wrist-worn  wearables,  particularly  products  like  the  Apple  Watch  would  be  an
accessibility chokepoint for most people, as few can afford it, especially in Malaysia.
Furthermore,  Train  Fitness’  solution  only  encapsulates  motion  tracking  and  rep
counting, without the capability to evaluate or judge the movement of the exercise. This
extends  to  most,  if  not all wrist-worn  wearables tracking.  This  is  simply  due  to  the
limitations of data collection capabilities of wrist-worn wearables, which collects data
from biomechanical readings. Hence, this path of wrist-worn wearables might be useful
for some, but it would not be able open the way forward in regard to the evolution of
telehealth and fitness technology. It is critical that a solution that is not only capable of
detecting  but  also  provide  accurate  evaluation  would  be  the  next  step  to  eventually
replace or rather replicate the expertise of human fitness experts.
It is obvious that in order to achieve this, a solution involving a camera is needed to
achieve this. Such camera would also need to be provided the appropriate processing
power  to  run  the  necessary  AI  model  capable  of  not  only detect  but also  evaluate
exercises. To replicate low latency feedback of wrist-worn wearables, it would be ideal
if  the  camera  is  supplied  with  local  edge  computing  processing. This  is  where
ExeVision emerges as a comprehensive and intelligent solution. ExeVision is proposed
to  be  a  camera-based  AI  system  that uses computer  vision  and  edge  computing  to
deliver low  latency exercise evaluation  and  corrective  feedback  without  the  need  for
wearables. By integrating a Luxonis OAK-D camera with an NVIDIA Orin Nano, the
system performs on-device pose estimation and form assessment using deep learning

CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 3
models  optimized  for  low-latency  inference.  Beyond binary detection,  ExeVision
emphasizes evaluation, translating  biomechanical  deviations that  often  required  a
human’s intuitive judgement into quantitative form scores. The incorporation of local
edge   processing   ensures   smooth,   uninterrupted   feedback   even   without   cloud
dependency, while a cloud mode extends functionality to remote analysis and progress
tracking.
## 1.2 Problem Statements
Human movement, specifically in  the  context  of  exercise,  is  deeply  nuanced,  and its
quality  depends  not  only  on  visible  motion  but  also  on  intent,  rhythm,  control,  and
biomechanical alignment. A human expert does not merely observe where body parts
are located in space; they evaluate the quality of the movement, analyzing factors such
as stability, rhythm, and safety margins to provide semantic feedback.  A system that
can observe, evaluate, and provide corrective insight with the same analytical reasoning
as a human coach would therefore mark a crucial step forward in bridging the divide
between human expertise and artificial intelligence in fitness technology (Zheng et al.,
## 2023).
A common   limitation of   current   solutions is   the inability   to   perform   genuine
quantitative   evaluation or   generate   actionable,   visually   grounded   feedback. For
example, work done by Mishra et al. (2025) and Chen and Yang (2020) demonstrate
posture detection, but these systems rely on binary classification, labelling movements
as “correct” or “incorrect”, without  quantifying  error  severity  or  providing  specific
corrective guidance. This is because these solutions rely on pattern matching. A neural
network  trained  solely  on labelled "correct"  versus  "incorrect"  squat  videos  may
correctly classify a test video, yet it cannot explain why the squat is deficient or which
specific action that caused the error. This phenomenon is similar to why an LLM tend
to hallucinate, as it only matches the pattern on the data it trained on, without the world
understanding  of  a  human. This  lack  of  interpretability  and  granularity  renders  such
systems  unsuitable  for  coaching  applications  where  users  need  concrete,  targeted
corrections rather than vague labels.



CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 4
Another problem is there is a disproportionate lack of personal trainers for gym goers
that leads to them to essentially doing the exercises ‘blind’. While they can refer and
see the videos of  the correct form, a second set of eyes, usually a personal trainer would
be  a  game  changer to  their  motivation.  Indeed, Radhakrishnan  et  al.  (2020)  identify
"lack of access to personal trainers" as a primary driver of dropouts among gym goers.
However, it  is logistically impossible  for everyone to  have  access  to  a  professional
personal  trainer.  In  fact,  only  10%  of  gym  goers have  registered  and  have  access  to
private  personal  training  (Lu,  Y et  al,  2024).  This  problem  is  especially  apparent  in
Malaysia, where exercise culture is still low, thus that percentage would be even lower.
Thus, a product or solution that can mimic human personal trainers, providing real-time
corrective  cues,  expert-like  feedback  on  exercise  form could  significantly  improve
exercise safety, adherence, and outcomes for the vast majority of gym-goers who train
without professional guidance.
While the lack of personal trainers in gyms and fitness centers is concerning, there are
those  who  do  not  even  have  access  to gyms  or consciously avoid it.  This  typically
happens in beginners, who are self-conscious, intimidated or simply too shy to exercise
in front of others, which is one of the main barriers of exercise (Gjestvang et al., 2020).
These people then do the next best thing, which is to train alone at home. However, this
further exacerbates the problem of improper form, as there is no one around them to
notice and correct any mistakes being made. This will lead to a reduced effectiveness
in whatever exercise is being done, thus reducing the results produced, or worse, face
increased risks of injuries. Thus, a safe, judgment-free channel for technique feedback
is critical if these self-conscious beginners are to benefit from exercise at all.
In summary, the main problem centers on mimicking the human evaluative intelligence
that  differentiates personal trainers  from  today's  AI-powered exercise  evaluation
solutions that can  track  motion, yet lack  the biomechanical reasoning, contextual
understanding,  and  interpretability  needed  to  judge  the  nuanced  quality  of  form  in  a
truly meaningful, human-like way. This shortfall is further compounded by the palpable
shortage of personal trainers, the economic and logistical barriers that keep most gym-
goers and home users from accessing expert-level feedback and reducing the reliance
on  costly  or  specialized  hardware  that  curtails  the  reach  and  effectiveness  of  AI-
powered exercise evaluation.

CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 5
## 1.3 Motivation
This  project  is  driven  by  both  personal  experience  and  observed  challenges  faced  by
new  gym-goers.  When  I  first  began  training  at  the  gym,  I  noticed  that  many  of  my
friends struggled to start or sustain their fitness routines, particularly those without gym
partners or prior experience. They were often unsure whether their exercise form was
correct, leading to frustration, inconsistent progress, or even minor injuries. Watching
online  tutorials  or  fitness  videos  could  only  help  to  a  certain  extent, these  resources
demonstrate the movements but cannot observe or correct a user’s mistakes in real time.
Hiring a personal trainer (PT) remains the most effective way to receive direct feedback,
yet it is financially inaccessible for many individuals, especially students or casual gym
users. As  a result, a large group of people attempt to learn independently, relying on
guesswork or external advice without knowing whether they are performing exercises
safely or effectively. This experience revealed a pressing need for a digital assistant that
mimics the role of a trainer, offering guidance, correction, and motivation, without the
recurring  cost  or  dependence  on  another  person. By  developing  a  camera-based
evaluation system that can analyse,  rate,  and provide corrective feedback on posture,
the project aims to empower individuals to train safely and confidently. It represents a
step  toward  bridging  the  gap  between  professional  coaching  and  everyday  fitness
through the power of intelligent computing.


CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 6
## 1.4 System Objectives
- To develop a Neuro-Symbolic AI model that generates 0–100 quantitative Form
Scores for ≥2 exercises, capable of detecting ≥3 specific errors per exercise, and
generating  actionable  feedback  via  dynamic  text  templates  +  3-frame  visual
evidence GIFs
- To engineer a low-latency Edge Computing module using Luxonis OAK-D and
NVIDIA  Orin  Nano  delivering  real-time corrective cues within  ≤1  second
latency and generating complete session reports within ≤1 minute after video
receival.
- To  deploy  a  serverless  Cloud  Web  Application  that  processes  videos with
size≤50MB asynchronous,  handles  ≥3 concurrent  uploads,  and generating
complete session reports within ≤2 minute after video receival.
## 1.5 Proposed Solutions
The  proposed  solution  is  ExeVision, AI-Powered  Camera  System  for  Evaluation  of
Exercises, developed both for on-premises in gym facilities and remotely through the
cloud. As illustrated in Figure 1 and Figure 2, ExeVision will be able to serve through
two  deployments,  an  Edge  Mode  and  Cloud  Mode.  The  Edge  Model  will  utilize  a
powerful depth sensing camera to provide a more accurate exercise evaluation, and the
Cloud Mode enhances the flexibility of the system to function remotely with the trade-
off of slightly lower accuracy due to the monocular 2D smartphone camera, though the
underlying AI pipeline is identical.By doing so, the solution aims to improve exercise
execution quality both in the gym, and for gym goers without access to the facilities.




CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 7



Figure 1: Flow of Cloud Mode





Figure 2: Flow of Edge Mode

CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 8
## 1.6 Project Module

## Figure 3: Project Modules


CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 9

Figure 3 shows how the ExeVision project is split into modules and specific features.
From a high-level overview, ExeVision is comprised of three main modules, which are
the  Edge  Processing  Module,  the  Web  Application  Module,  and  of  course,  the  AI
Evaluation   Module.   The   Edge   Processing   Module   particularly focuses   on   the
implementation of the Edge Mode deployment using Luxonis OAK-D and Jetson Orin
Nano. In particular, since the Jetson Orin Nano runs on Ubuntu, a certain configuration
needed to be set up and developed. This module also encompasses the setting up of a
local AI server to serve as the backend, which is using FastAPI. A local Web GUI will
also need to be set up, though it will almost entirely be re-used from the Web App GUI.
A local database such as SQlite will also be developed as the local database
For the Web Application Module, it mainly serves Cloud Mode deployment. The same
GUI will be deployed on a hosting site, such as Vercel. It will then be supplemented
with the Database, Storage and Backend services like Authentication. For the AI, it is
necessary  to  have  a  serverless  solution  that  can  host  and  run  the  custom  AI  Model
instead  of  running  locally  like FastAPI.  Thus,  a  cloud  AI  server  will  be  set  up  to
communicate with the web app host in order to process the videos.
Finally, the AI Evaluation Module concerns the development of the AI Model common
to both deployment modes. It  can then be split further into the Feature Extractor and
Normalization component which  extracts  human key  points and  landmarks  from  the
video. Furthermore, A State-Aware Segmenter then divides the movement stream into
idle,  eccentric,  and  concentric  phases,  rejecting  clips  that  fail  the  front/side  view
validation. Biomechanical Rule  Scoring  applies  micro-programs  to  detect  specific
faults and  quality such  as knee  valgus,  insufficient  hip  depth, joint  angles,  etc.  A
BiLSTM  Neural  Scoring  network evaluates rhythm  and  control  while  the  Score
Aggregator aggregates symbolic  and  neural  signals  into  a  single  0–100  Form  Score.
The Feedback Engine populates dynamic templates with error details, retrieves 3-frame
GIF evidence, and returns actionable coaching text, all without resorting to generative
AI that could hallucinate unsafe advice.

CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 10
1.7 Benefits / Impact / Significance of Project
This project delivers practical and societal value by introducing an intelligent exercise
evaluation  system  that  is  immediately  usable  by  non-technical  users.  Designed  as  an
out-of-the-box solution, it removes the typical barriers associated with research-based
or  developer-only  fitness  systems.  Users  can  simply  connect  the  device,  calibrate
briefly,  and  begin  exercising  with  real-time  corrective  feedback.  This  accessibility
supports  the  wider  adoption  of  AI-assisted  exercise  guidance  in  homes,  schools,  and
fitness centres, where professional supervision may not be available.
The system also provides meaningful public health impact by promoting safer exercise
habits.  Through  posture  assessment  and  instant  correction,  it  helps  prevent  common
form-related  injuries  and  ensures  users  perform  exercises  more  effectively.  For
beginners  and  individuals  without  access  to  personal  trainers,  it  serves  as  a  reliable
digital  companion  that  reinforces  good  training  discipline.  Beyond  fitness,  the  same
technology  can  be  adapted  for  rehabilitation  and  physiotherapy,  supporting  broader
healthcare applications.
From  an  academic  and  technological  perspective,  the  project  strengthens  the  link
between  intelligent  computing  research  and  human  well-being.  It  demonstrates  how
computer  vision  and  IoT  can  be  combined  to  address  real-world  health  challenges
through  affordable  innovation.  Its  contribution  aligns  with  SDG  3:  Good  Health  and
Well-Being,  as  it  encourages  consistent  physical  activity,  accessibility,  and  safe
exercise  practices  across  different  user  groups. This system  transforms  intelligent
exercise  monitoring from  a  research  concept  into  a  functional,  everyday  technology
with measurable benefits to users and society.


CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 11
1.8 Uniqueness of Proposed Solutions
ExeVision  stands  out  among  the  current  exercise  evaluation  solutions  thanks  to  a
number of innovative features that aims to closely mimic the nuanced evaluation of a
living and breathing human personal trainer  Compared to other similar Action Quality
Assessment   (AQA) solutions,   ExeVision   integrates   a Hybrid Neuro-Symbolic
Architecture,  enabling  the  system  to  evaluate  exercises  based  on  both  biomechanical
safety  rules  and  movement  smoothness.  This  feature  is  especially  valuable  in  self-
guided  training  settings,  where  binary  classification  (correct  vs.  incorrect)  fails  to
capture   the   nuance   of   human   movement.   Through   the   combination   defined
biomechanical   rules   with   neural   networks,   ExeVision   promotes   precise   error
quantification and a more comprehensive judgement.
ExeVision’s feedback generation capability, facilitating the delivery of actionable and
reality-grounded directives and visual evidence, acts as an enhancement compared to
existing works that rely on vague confidence scores or generic alerts, which lacks the
context  and  clarity  users  need  for  specific  corrections.  By  introducing  Dynamic
Template  Slot-Filling  and  visual  GIF  retrieval,  ExeVision  fosters  understanding  and
remediation, allowing users to feel more guided and corrected.
Furthermore, ExeVision stands out because it is one of the only, if not the only solution
that aims to provide a B2B solution for gym facilities by using 3D-depth sensing camera
such as the Luxonis OAK-D and a portable edge computing of NVIDIA  Jetson Orin
Nano.  Most  solutions,  especially current deployed  solutions are  built  as  a  mobile
application. This means  that  the  video  is  streamed  on  a monocular 2D  smartphone
camera which inherently lacks depth perception, resulting in lower accuracy especially
in the nuanced evaluation of fitness exercises. In contrast, the OAK-D’s stereo RGB-D
pipeline delivers millimeter-level depth maps at 120 fps, while the Jetson Orin Nano’s
1024  CUDA  cores  and  32  Tensor  Cores  run  the  entire  hybrid  AI  stack  locally,
eliminating  cloud  latency  and  preserving  user  privacy. While Cloud Mode  does  not
offer  the  OAK-D’s  depth  sensing  capabilities,  it  still  uses  the  same  innovative  AI
pipeline and can be used remotely from anywhere. Thus, unlike others who only offer
cloud and 2D based solutions, ExeVision offers the best of both worlds.



CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 12
1.9 Organization of Report
This  analysis  report  consists  of  4  main  sections.  The  first  section  discusses  the
introduction,  which  involves  the  project  background  and  problem  statements  to  be
solved. The  system  objectives  and  modules  are  briefly  discussed,  followed  by  the
benefits  of  the  project  and  the  uniqueness  of  the  proposed  solution. Section  2  then
discusses the background and related work, which involves the project status, project
context,  comparison  analysis  of  similar  systems,  and  finally  introduction  of  the
proposed  solution. Section  3  discusses  the  system  requirements  and  analysis,  which
involve  the  project  scope,  project  capabilities  and  limitations,  project  management,
development methodology, SWOT analysis, and the detailed requirements of the new
system. Section  4  details  the  system  design  and  implementation,  which  involves  the
diagrams  included  in  the  system  design,  detailed  descriptions  of  the  project,  and  the
intelligent methods used. Lastly, the conclusion and SDG alignment sections wrap up
the report.













CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 13
## 2 BACKGROUND AND RELATED WORK
2.1 Status of the Project
This project should be classified as an enhancement of existing projects, as there have
been  several  similar  projects  that  aim  to  provide  human  judgement  of  exercise  to
computer vision. However, they each have their own approach, and almost all, if not all
are still in the research stage. Widespread commercialization of this technology has not
happened yet.
## 2.2 Project Context
This  project  can  be  considered  to  be  an  Action  Quality  Assessment  (AQA)  Project.
AQA projects aims to  evaluate the quality of performed  actions, offering a different,
more objective evaluation compared to human assessments that are naturally subjective
(Zhou,  K.  et  al,  2024). A  typical  logic behind  AQA follows  a  2-stage  pipeline  (see
Fig.4), which is  to first extract  key  movement  features  of  the  human  body  using
computer vision, and based on certain rules, patterns or ranking, map them into a score
that reflects the quality of the movement (Parmar, P., & Tran Morris, B., 2017).

Figure 4: Generic AQA Pipeline
The mapping of human features has been done in several ways. First, Regression-based
scoring,  where  the  model  predicts  a  numerical  score  on  a  continuous  scale,  using
models such as linear  regressors,  shallow/deep  neural  nets,  support  vector  regressors
(SVR), and  more  (Yin,  H.  et  al,  2025). Next, Classification-based scoring,  where

CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 14
Instead of a continuous number, the model assigns the action to a discrete category or
label  (e.g.,  "Good,"  "Fair,"  or  "Poor"). Another  approach  is Pairwise  ranking. This
technique compares two or more action samples to determine which has higher quality.
Lastly, Rule-based   scoring,   where   unlike statistical   methods   that   approximate
relationship,  incorporates  human  expert  knowledge  of  biomechanical  standards  to
handcraft a set of rules that can compute the action quality score.
Among AQA projects, there are certain levels of complexity. The simplest AQA is an
AQA project that can map extracted features to a certain value or score, but unable to
explain why it derived the score. Thus, the next level of AQA complexity can progress
beyond scoring and dig deeper into the why and how of such scores. It can assess the
extracted  features,  analyze  and  determine  the  severity  of  each  detected  pattern.  The
current  highest  complexity  of  AQA  models  is Feedback AQA,  where  besides
determining quality, can also suggest what part can be improved. Hence, ExeVision in
this case aims to become a Feedback AQA, where the quality of gym exercise can be
analyzed and provide useful feedback to the user. (Yin, H. et al, 2025).
2.3 Comparison Analysis of Similar Projects
a) Project A: Learning to Assess Squat Technique: Video-Based Pose Analysis
with  Classical  and  Deep  Learning  Models  (Hjaltason,  M.,  &  Gertrud,  U.
## (2025).)
Several  contemporary  projects  exist,  not  only  in  the  AQA  domain,  but
specifically in the fitness domain. For example, a student thesis from Linnaeus
University, Sweden sought to  investigates the use of machine learning models
for  classifying  squat  technique.  In  this  thesis,  they  used  two  popular  machine
learning  models,  which  are Random  Forest  and  a  Bidirectional  Long  Short-
Term Memory  (BiLSTM)  model,  for  classifying  squat  technique  from  video-
based  pose  data.  Using  a  custom  dataset  of  252  squat  videos,  the  authors
identified  that  both  models  are  viable  option  for  identifying  correct  squat
techniques,  but  Random  Forest  generally  performed  better,  with  a  97.35%
accuracy, and BiLSTM having 94.61% accuracy of 94.61.

b) Project   B: Pose   Trainer:   Correcting   Exercise   Posture   using   Pose
Estimation (Chen, S., & Yang, R. R. (2020)

CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 15
Another  relevant project is  Pose  Trainer:  Correcting  Exercise  Posture  using
Pose   Estimation by   Steven   Chen   and   Richard   R.   Yang   from   Stanford
University. Here, beyond the module trainer of the previous project, they built
a desktop application designed to analyze exercise posture using pose estimation
and  provide  form-related  insights. In  this  project,  they  shifted  from  the  pure
machine  learning  models,  but  incorporated  geometric heuristic rules,  such  as
calculating the angle between joints as a benchmark. They then compared it with
a machine learning pipeline using Dynamic  Time  Warping  that  are  trained  to
perform automatic  classification  of  correct  versus  incorrect  form  across  the
recorded  dataset (100  videos).  The  authors concluded that the  geometric
heuristic approach is superior for providing personalized feedback, with a 100%
accuracy  on  Good  Videos  and  Bad  Videos,  and DTW/ML  approach  is  more
suitable for detecting  overall  correctness,  but  less  suitable  for  generating
feedback to the user.
c) Project  C: Domain  Knowledge-Informed  Self-Supervised  Representations
for Workout Form Assessment (Parmar, P., Gharat, A., & Rhodin, H. (2022,
## October))
This  project  takes  quite  a  different  approach  compared  to  other  projects.  In
contrast to systems that rely on pose estimators or handcrafted geometric rules,
this   project   focuses   on   learning   exercise-oriented   visual   and   motion
representations through self-supervised training, enabling the system to detect
workout-form  errors  without  depending  on  pose  extraction.  The  authors
designed  two  domain-informed  Self-Supervised  Learning  methods,  namely
Pose   Contrastive   Learning   and   Motion   Disentangling,   that   leverage the
common  motion  of  exercises  (e.g.,  squat  descent  and  ascent  cycles)  and
synchronized  barbell  trajectories  to  learn  features  that  naturally  encode  both
global self-supervised and local anomalous motions such as knee valgus or torso
lean. After self-supervised pretraining, these representations are fine-tuned on a
labelled  subset  of  their  dataset.  The  reason  why  this  project  is  different,  is
instead of generating feedback based geometric calculations, the system outputs
multilabel  error  probabilities  for each  exercise,  essentially  indicating  which
specific form faults occurred and how likely they were, such as “knees inward,”
“insufficient depth,” or “lumbar rounding. They concluded that their project is

CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 16
more robust to real-world gym conditions, more sensitive to subtle errors, and
more scalable than pose-dependent methods.
d) Project D: Gymscore
Gymscore is a smartphone-based AI-powered form analysis, progress tracking,
and personalized coaching app. From its official website, it advertises itself to
be able to provide objective scores of technique quality, identify issues, assess
risk  of  injuries  and  track  progress  over  time. Due  to  it  being  a  commercial
project,  the  underlying  AI  model  is  undisclosed,  with  no  technical  details
explaining  how  it  analyses exercises. Currently,  it  is  predominantly  a  B2C
(Business-to-Consumer) business, with the main offering being  a paid mobile
app where users upload videos and receive their analysis reports. They also offer
SDK and API for coaches to use their technology on their own platforms. From
this, it can be inferred that they rely on cloud processing to finish their analysis
pipeline, since both the mobile app and the API require an upload step before
any feedback is returned.

2.4 Related Projects’ Strength and Weaknesses
a) The project exhibited a strong accuracy in correctly classifying the correct squat
form  based  on  the  limited  dataset  it  is  trained  on.  It  identified  that machine
learning  models  like Random  Forest and BiLSTM are  capable  of  classifying
squat  techniques,  with  RF  being  superior  in judging correct biomechanical
features  of  squatting,  while  BiLSTM is  better  in capturing temporal  motion
structure, or the overall smoothness of the movement. However, the model does
not  offer  any  more  beyond  binary  classification.  The  two  models  used  are
inherently  non-deterministic  and classifying model,  so  they  are  unable  to
interpret  the  extracted  features  to  provide  a personalised feedback  or  error
analysis. The project also did not move beyond the model and research phase,
with no real-world deployments being done on any platforms.
b) Project  B tested  with a  heuristic-based  approach  that  enables  specific  error
identification  and  generates  personalized  feedback  by analysing geometric
relationships  between features.  The  authors  also  deployed  the  model as  a
functional  desktop  application which demonstrates  practicality.  The  body-
normalized  metrics  represent  a  significant  improvement  over  unnormalized

CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 17
methods used in Project A, adapting to different physiques and camera distances.
However,   the   desktop app deployment   proves   inconvenient   for   gym
environments  as  acknowledged  by  the  author,  where  they  admit  that  a
smartphone-accessible  solution  would  be  one  avenue  of  improvements  for
future works., Furthermore the purely heuristic rules lack subjectivity, as relying
on  rigid  thresholds  makes them ‘brittle’ and  unable  to  adapt  to  individual
anatomical variations. While DTW adds data-driven flexibility, it operates as a
black  box  without  explanatory  capabilities,  leaving  users  without  corrective
guidance when misclassifications occur.
c) Project C presented a novel and creative approach to AQA. Instead of judging
from the pose exhibited by the features extracted, their model evaluates it from
the barbell’s movement to see how ‘smooth’ the overall movement is compared
to  the  benchmark.  This  substantially  improves  the  robustness  of  the  model  to
video environments , as it can ignore camera angles, obstruction or visibility of
joints. However, the model is still at its core a classifier, and unable to perform
any generative feedback. It cannot provide how the user can improve or mitigate
their  form  issues. The  system also only  operates  offline  and  does  not  support
real-time  analysis  or  instant  correction,  limiting  its  applicability  for  live  gym
coaching or interactive fitness systems
d) Project  D,  Gymscore  stands  out  from  the  rest  of  the  previously  mentioned
projects,  due  to  the  fact  that  it  is  an  actual,  commercial  mobile  application
available  to  download  and  use  now.  This  signifies  a  major  step  from  being  a
research paper in academic circles to actual deployment, making Gymscore one
of  the  earliest  players  in  the  market  for  exercise  AQA  apps.  Gymscore  offer
capabilities  that  improve  upon  the  binary  capabilities  of  other  mentioned
projects,   delivering   objective   0   to   100   scores,   progress   tracking,   and
personalized  coaching  through  a  polished  UI/UX.  They  also support many
exercises, with 500 being the claimed figure. However, their nature as a closed-
source, proprietary model leads to an unclear understanding of how exactly the
analysis  works.  Early  reviews  have  cited  the  analysis  to  be  inconsistent,
potentially due to the black-box and probabilistic  nature of the underlying  AI
used by Gymscore. There is also no mention of any preprocessing being done
on  the  video  input,  such as  normalization,  temporal  segmentation  of  different
phases,  or  view  validation.  Lastly,  using  generative  AI  for  coaching  poses  a

CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 18
safety risk because, in contrast to deterministic rule-based systems, probabilistic
models  may  "hallucinate"  or  produce  advice  that  sounds  reasonable  but  is
biomechanically incorrect.
## Features
## Project A
(Hjaltason
## & Uwaoma,
## 2025)
## Project B
(Chen &
## Yang,
## 2020)
## Project C
(Parmar et
al., 2022)
## Project D
(Gymscore)
## Video-based
## Assessment
## ✔ ✔ ✔ ✔
3-D Depth Sensing.
## ✗ ✗ ✗ ✗
## Pose Estimation
## Required
## ✔ ✔ ✗ ✔
## Temporal Motion
## Modelling
## ✔ ✗ ✔ ❓
## Personalized
## Feedback
## ✗ ✔ ✗ ✔
## Interpretable Error
## Explanation
## ✗ ✔ ✗ ✔
Robust to Camera
## Angle & Occlusion
## ✗ ✗ ✔ ❓
## Multi-label Error
## Detection
## ✗ ✗ ✔ ✔
Real-Time
## Corrective Cues
## ✗ ✗ ✗ ✗
Deployed as End-
## User Application
## ✗ ✔ ✗ ✔

Table 1: Comparison of Current Solutions


CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 19

2.5 Brief Introduction of Proposed Solution
The review and comparison of some of the current implementation of a smart exercise
evaluation reveals several common trade-offs. They are either highly interpretable but
inflexible, or essentially black boxes without knowledge of biomechanical mechanisms.
Thus,  ExeVision  aims  to  address  these  pain  points  and  deliver  an  improved  system
designed  to provide  a  smarter,  yet  grounded  evaluation  of exercises  to  fitness
enthusiasts.
ExeVision offers  the  dual  capabilities  of  the  deterministic biomechanical  rules
alongside neural models such as Bi-LSTM to provide a unified and aggregated score
that  combines  the  best  of  both  worlds. ExeVision is offered in  two  complementary
modes. First, the edge configuration that performs real-time inference on an NVIDIA
Jetson  Orin  Nano  coupled  with  a  Luxonis  OAK-D camera and  a  serverless  cloud
configuration  that  asynchronously and  concurrently analyzes  smartphone exercise
videos of users.












CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 20
## 3 SYSTEM REQUIREMENTS AND ANALYSIS
3.1 Project Scope , Capabilities and Limitations
## 3.1.1 Project Deliverables
- Edge-hosted local web application interface deployed on NVIDIA Jetson Orin
## Nano.
- Serverless cloud web application for remote video analysis
- AI model capable of generating quantitative quality assessment
- Visual evidence retrieval for extracting peak-error GIF snippets.
- Natural language generation  engine  for  dynamic,  interpretable  corrective
feedback formulation.
## 3.1.2 Project Exclusions
- Automatic exercise discovery
- Multi-person tracking or simultaneous analysis of multiple subjects
- Clinical rehabilitation diagnosis or medical physiotherapy tracking
## 3.1.3 Project Stakeholders
- Individuals performing resistance exercises
- Professionals responsible for refining the biomechanical rules for the AI Model.
- Gym owners that  deploy  the  ExeVision  system  in  Edge  Mode  to  provide  a
value-added service to their members.
## 3.1.4 Project Capabilities
- Extract human pose landmarks using MediaPipe’s Pose Landmark Detection
## Model
- Perform geometric normalisation to scale landmark data relative to user torso
length.
- Implement  automated  view  validation  to  detect  and  reject  invalid  camera
perspectives.

CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 21
- Perform  state-aware  temporal  segmentation  to  distinguish  movement  phases
based on velocity.
- Execute  Neuro-Symbolic  analysis  to  identify  form  violations  and  generate
smoothness scores.
- Generate  interpretable  feedback  through  real-time  visual  cues  and  natural
language prompts.
## 3.1.5 Project Limitations
- Exercise offered is limited to manual creation of the biomechanical rules.
- Accuracy is sensitive to environmental conditions such as low lighting or poor
visual contrast.
- Functionality is limited to single-user analysis without support for multi-person
environment

CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 22
## Project Management
## 3.1.6 Work Breakdown Structure


CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 23
## 3.1.7 Gantt Chart
## TASK OCT NOV DEC JAN FEB MARCH APRIL MAY JUNE
## 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4
## PROPOSAL
## Project Bidding
## Write Project Proposal
## Check Project Proposal
## Submit Project Proposal
## SYSTEM  REQUIREMENT  &
## DESIGN

## Proposal Feedback
Research on Project Proposed
## Identify Project Requirements
## Design Required Diagrams
## Writing Requirement Report
## Checking Requirement Report
## Submit Requirement Report
## Requirement Report Presentation
## Preparation

## Requirement Report Presentation
## PROTOTYPE
## Prototype Design
## Prototype Development
## Prototype Testing
## Prototype Presentation
## Preparation

## Prototype Presentation
## FINAL SYSTEM
## System Development
## System Testing
## Write Final Report
## Check Final Report
## Submit Final Report
## Final System Presentation
## Preparation

## Final System Presentation
## PIXEL

CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 24
## 3.1.8 Milestone Timeline
Phase Deliverable Tasks involved Deadline
## Requirements
gathering
## Initial Project
## Proposal
- Feasibility study and
market validation
- Comparison analysis
of similar systems
- Proposal submission
## 26/10/2025
System design System    Requirement
and    Design    (SRD)
## Report
- Architecture diagram
design
## •  Database  &  Class
diagram design
## •    Flowchart    &
Algorithm design
-  Use  case  diagram
design
-    SRD    Report
preparation
## 15/12/2025
SRD Presentation SRD Presentation
## Slides
-  Presentation  deck
preparation
-  Milestone  1:  SRD
Presentation delivery
## 21/12/2025
## Prototype
## Development
## Project Progress
## Demo
•Core feature
implementation
•Prototype
development
## • Milestone 2: Project
## Progress Demo
## 29/03/2026
## System
## Implementation
Source code and
infrastructure
- Full code
development
•Cloud/Edge
infrastructure and
hosting setup
- System integration
## 24/05/2026
## System  Testing  &
## Deployment
## Final Report
(Softcopy)
- Unit, Integration, and
System testing
-   User   acceptance
testing (UAT)
- Final Report  writing
and submission
## 24/05/2026
## Final System
## Presentation
## System
## Demonstration
-  Final  presentation
preparation
## •  Milestone  3:  Final
## System Demo
## 07/06/2026


CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 25
## 3.2 Development Methodology
The    ExeVision    Project    follows    an Iterative    and    Incremental    Development
methodology.  The  project  will  focus  on  working  software,  with  an  initial  prototype
developed early but without most of the features. Over time, more and more features
will be added towards the final implementation
The  development  lifecycle starts with  the UI/UX  aspect. In  this  first  increment,  the
focus  is  strictly  on  constructing  the  "base"  of  the  system,  for  example implementing
user  login,  homepage,  video  uploads,  and  the user dashboard.  At  this  stage,  the
application functions as a static prototype with mock data and placeholders. This allows
for critical early validation of the user journey, knowing what is needed, thus providing
a platform for future iterations to progress from
Once the interface design is verified, the second increment addresses Data Persistence
and  Core  Backend  Integration.  This  phase  transforms  the  static  prototype  into  a
dynamic application capable of memory. A database structure is integrated to store user
profiles, session logs, and history. The application is updated to communicate with this
storage layer, enabling functionalities such as registering users, saving video files, and
retrieving past performance data.
The system acquires its first layer of analytical capability in the third increment. During
this  phase,  Computer  Vision  and  Basic  Symbolic  Analysis  are  integrated.  The  pose
estimation  module  extracts  skeletal  landmarks  from  video  frames,  while  an  initial
Symbolic   Rule   Engine   detects   fundamental   biomechanical   errors   such   as   joint
misalignment.   This   phase   also   involves   implementing   the   Decision   Feedback
Generation  module,  which  translates  raw  data  into  actionable  coaching  feedback.  A
Dynamic Template Slot-Filling technique populates text templates with specific error
details,  and  a  Visual  Evidence  Retrieval  algorithm  extracts  relevant  frame  sequences
illustrating detected faults. The application is updated to replace its placeholder content
with these live scores and visual cues, evolving into a functional form-assessment tool
The  final  increment  represents  the  maturation  of  the  system  where  it  truly  becomes
"smart." Advanced logic is added to the AI model, shifting the system's capability from
simple error detection to comprehensive performance evaluation. This involves training
neural networks to recognize complex temporal nuances that static logic cannot catch,

CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 26
such  as  movement  rhythm,  stability,  and  control.  Concurrently,  the  Symbolic  Rule
Engine  is  extensively  upgraded  with  a  broader  set  of  biomechanical  rules  and
sophisticated  rubrics,  enabling  the  system  to  enforce  stricter  and  more  nuanced
standards across various error patterns. By fusing this expanded library of rigid rules
with  the  neural  network's  intuition,  the  system  generates  a  unified  quality  score  that
effectively replicates the expert judgment of a human coach.
3.3 SWOT Analysis
## STRENGTHS WEAKNESS
- Combines deterministic
biomechanical rule-based
reasoning with Bi-LSTM neural
networks for temporal movement
analysis.
- Outputs 0 to 100 technique
quality score
- No black-box limitation of
purely statistical neural model
- Use defined text templates to
explain errors
- Provides 3-frame visual evidence
of error
- Dual-mode deployment, edge
mode for local processing and
serverless cloud mode for
concurrent analysis
- Relies on hand-crafted
biomechanical rules per exercise
- Extensive and slow exercise
expansion process
- Requires specialized hardware
(OAK-D and Jetson Orin Nano)
- Only have limited supported
exercises
## OPPORTUNITIES THREATS
- B2B gym partnerships
- Reduce trainer workload through
automated form checking
- AQA framework adaptable to
physiotherapy, injury
rehabilitation or even athletic
sports  training
- Existing platforms offer polished
user experiences and large user
bases.
- Potential erosion of the hybrid
architecture’s competitive
advantage without continuous
innovation.
- Dependence on specific sensors
and edge-compute devices.
- Camera-based body-data
analysis raises compliance
concerns (e.g., GDPR)


CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 27
## 4 SYSTEM DESIGN AND IMPLEMENTATION
## 4.1 Diagrams
## 4.1.1 Use Case Diagram

Figure 5: ExeVision Cloud Mode Use Case Diagram


Figure 6: ExeVision Edge Mode Use Case Diagram



CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 28
## 4.1.2 Use Case Diagram Description
Use Case ID UC-001
## Use Case Name Create Account
## Actor Guest
Scenario A new, unregistered user signed up for an ExeVision account.
Triggering Events Guest clicks "Sign Up" on the ExeVision web app.
Descriptions This use case enables new users to create an account in the
ExeVision web app. The account serves as the container for
all  session  data,  saved  evaluations,  and  progress  tracking
across multiple workouts.
Pre-Conditions
- Guests must have a valid email address.
- Guests must use an email address that is not already
registered to the system.
Post-Conditions
- User details rows are created in Supabase
- System    sends    verification    email    for    account
activation
Flow of Activity
- Guest navigates to registration form
- Guest enters required information and submits
- Guest verifies email via verification link
## Alternative Flow
- Invalid    Input:    If    email    exists    or    password
requirements   fail,   system   displays   specific   error
messages and returns to step 1.
- Abandoned  Registration:  If  Guest do  not  click  the
verification link, account remains in pending state for
24 hours before cancellation.
Table 2: UC-001 Description

CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 29
Use Case ID UC-002
Use Case Name Log in to Account
## Actor Registered User
Scenario Returning user authenticates to access personalized exercise
evaluation dashboard and historical data.
Triggering Events User  clicks "Log In" on the ExeVision web app.
Descriptions This case   validates   user   credentials   against Supabase
authentication  service.  Upon  successful  login,  the  system
loads  the  user's  profile. The  authentication  token  grants
access  to  cloud  storage  for  video  uploads  and  personal
analytics dashboard.
Pre-Conditions
- User must have active, verified account
- Users submit correct credentials
Post-Conditions
- User session token is established (valid for 8 hours)
- User profile data is loaded into application context
Flow of Activity
- User enters email/password  and clicks log in.
## Alternative Flow
- Account  Locked: After  5  failed  attempts,  account
locks for 30 minutes. System displays lockout timer
and support contact.
Table 3:UC-002 Description






CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 30

Use Case ID UC-003
## Use Case Name Forgot Password
## Actor Registered User
Scenario Users who have forgotten  their  password request a  secure
reset link to regain account access.
Triggering Events User clicks "Forgot Password" on login page.
Descriptions This  use  case  enables  password  recovery  via  email.  The
system generates a time-limited, single-use token and sends
it to the registered email address. The token links to a secure
page where the user can set a new password, regaining access
to their profile.
Pre-Conditions
- User account exists and is verified
- User has access to registered email
Post-Conditions
- Password  is  updated  and  encrypted  in Supabase
database
Flow of Activity
- User clicks "Forgot Password"
- User enters registered email
- User clicks reset link from email
- User enters new password and submits
## Alternative Flow
- Email Address not found : Display error and notify
user to use previously registered email.
Table 4: UC-003 Description




CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 31
Use Case ID UC-004
## Use Case Name Start Exercise Evaluation
## Actor Registered User
Scenario Users  initiate a  complete  cloud-based  analysis  workflow  to
evaluate  exercise  form  from  a  recorded  video to receive
quantitative scores and AI-generated feedback.
Triggering Events User  clicks "Start Analysis" from dashboard.
Descriptions This use case starts the cloud  evaluation  pipeline. Upon
completion,  it  presents  a  session  report  containing  scores,
error severity percentiles, visual evidence GIFs, and feedback
recommendations.
Pre-Conditions
- User is authenticated (session token active)
- User    has    recorded    video    files    (max    50MB,
MP4/MOV format)
- User  has  selected  target  exercise  from  supported
library
Post-Conditions
- Video files are stored in Supabase blob storage
- Upon  completion, app updates  to  calculated  Form
Score and Error Report page.

Flow of Activity
- User clicks “Start Analysis”
- User selects exercise to do.
- User uploads video.
## Alternative Flow
- Invalid Video: If view validation fails , or invalid file
type, system aborts and displays error.
Table 5:UC-004 Description


CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 32

Use Case ID UC-005
## Use Case Name View Dashboard History
## Actor Registered User
Scenario User reviews historical exercise sessions.
Triggering Events User clicks "Dashboard" tab upon login.
Descriptions This use case retrieves and visualizes aggregated session data
from cloud database. The dashboard displays a chronological
list  of  completed  evaluations  with  metadata:  exercise  type,
timestamp,  composite  Form  Score,  primary  error  type,  and
thumbnail of visual evidence.
Pre-Conditions
- User must be authenticated
- At  least  one  completed  session  must  exist  in  user's
history
- Cloud database must be accessible
Post-Conditions
- Dashboard  renders  with  session  records  sorted  by
date (newest first)
- Summary statistics are calculated and displayed (avg
score, total reps analyzed)
Flow of Activity
- User clicks “Dashboard”
- User views summary cards.
## Alternative Flow
- Empty  History: For  new  users,  system  displays
"Complete   your   first   evaluation   to   see   progress
tracking here
Table 6: UC-005 Description


CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 33

Use Case ID UC-006
## Use Case Name Upload Video
## Actor Registered User
Scenario User   transfers   exercise   video   file   to   cloud   storage   for
analysis.
Triggering Events User selects video file in "Start Analysis" flow.
Descriptions Sub-process of Start Analysis Evaluation. Handles secure file
ingestion, validation, and cloud storage.
Pre-Conditions
- User must be authenticated
- File is MP4/MOV 50MB.
- Video contains detectable human motion
Post-Conditions
- Session ID generated
- Processing task queued for evaluation
Flow of Activity
- User clicks file selector
- System validates format/size
- System uploads to cloud and extracts metadata
## Alternative Flow
- Format  Rejection:  System  displays  "Please  upload
MP4 or MOV"
- Size Exceeded: System prompts to compress file
Table 7:UC-006 Description





CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 34
Use Case ID UC-007
## Use Case Name Select Exercise
## Actor Registered User
Scenario User specifies which resistance exercise is in the video.
Triggering Events User chooses exercise in evaluation setup.
Descriptions Sub-process  of  Start  Exercise  Evaluation.  Loads  exercise-
specific   symbolic   microprograms   (biomechanical   rules),
camera  view  requirements  (Front/Side),  and  appropriate
BiLSTM model for smoothness scoring.
Pre-Conditions
- User authenticated
Post-Conditions
- Exercise ID bound to session
- Rule engine initialized
Flow of Activity
- User clicks exercise dropdown
- System displays supported exercise
- User selects exercise
- System loads rules and view requirements
## Alternative Flow -
Table 8: UC-07 Description







CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 35
Use Case ID UC-008
## Use Case Name View Evaluation Analysis
## Actor Registered User
Scenario User reviews AI-generated feedback and form scores.
Triggering Events Cloud processing completes.
Descriptions Core  feedback  delivery.  Presents  session-level  Form  Score
(0-100),  per-rep  error  analysis  (type,  severity  percentile,
frame  index),  BiLSTM  smoothness  score,  and  3-frame  GIF
visual evidence. Populates dynamic coaching templates with
specific corrections.
Pre-Conditions
- Session is ready.
- All analysis artifacts persisted
Post-Conditions
- Analysis report rendered in browser
- GIF assets loaded
Flow of Activity
- User redirected after video finish processing
- Page displays composite Form Score and per-rep
breakdown
- User clicks error GIF to replay key moment
- User optionally saves session
## Alternative Flow
- Dashboard Flow : User clicks ‘Show Details’ in
session card
- Analysis Failed: System displays error logs and
suggests re-upload.
Table 9:UC-008 Description



CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 36
## 4.1.3 Sequence Diagram

## Figure 7: Create Account Sequence Diagram






















































CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 37

## Figure 8: Log In Sequence Diagram







































CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 38

## Figure 9: Forgot Password Sequence Diagram



























































































CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 39

## Figure 10: Cloud Mode Sequence Diagram



CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 40

## Figure 11: Edge Mode Sequence Diagram



CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 41
## 4.1.4 Entity Relationship Diagram

Figure 12: Project ERD





CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 42
4.2 Detailed Description of Project
## 4.2.1 Functional Requirements
## No. Requirement Description
## 1.  User Registration
Allow new guests to create accounts with
email verification
-  User Login/Logout
Authenticate registered users and manage
8-hour session tokens
## 3.  Password Reset
Enable   password   recovery   via   secure,
time-limited email link
## 4.  Profile Management
Store  and  update  user  biometric  data  and
fitness level
## 5.  Exercise Selection
Let users pick from a supported resistance-
exercise library
## 6.  Video Upload
Accept  MP4/MOV  ≤50  MB  for  cloud
analysis
## 7.  View Validation
Detect     and     validate     camera     angle
(front/side) before processing
-  Real-Time Edge Analysis
Stream  and  evaluate  live  video  on Orin
Nano + OAK-D
## 9.  Temporal Segmentation
Auto-segment reps into
eccentric/concentric/idle phases
## 10.  Pose Landmark Extraction
Extract  2D/3D  skeletal  landmarks  with
body-normalized scaling
## 11.  Error Detection
Identify  form  violations  using  symbolic
biomechanical rules
## 12.  Severity Scoring
Assign   percentile   severity   (0–100)   per
error
## 13.  Neural Smoothness Scoring
Compute    movement    smoothness    via
BiLSTM model
## 14.  Composite Form Scoring
Fuse   symbolic   and   neural   scores   into
unified rep quality rating
-  Real-Time Visual Cues
Show color-coded boxes and text prompts
in edge mode

CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 43
## 16.  Feedback Generation
Populate  templates  with  error  details  for
readable coaching
## 17.  Visual Evidence
Extract  3-frame  GIFs  centered  on  peak-
error timestamps
## 18.  Session Report
Present  session  scores,  per-rep  analysis,
and visual evidence
## 19.  Dashboard History
List completed sessions with metadata and
thumbnails
## 20.  Progress Analytics
Calculate  summary  stats  (avg  score,  total
reps) over time
## 21.  Job Queue Management
Track video-processing status
## (queued/processing/completed/failed)
## 22.  Template Management
Store   dynamic   feedback   templates   per
error type/severity
-  Cloud-Mode Processing
Asynchronously analyse uploaded  videos
via serverless backend
## 24.  Data Persistence
Reliably store users, sessions, reps, errors,
and evidence

## Table 10: Functional Requirements




CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 44
4.2.2 Non-Functional Requirements
## No. Requirement Objective & Measurable Criteria
-  Real-Time Feedback
## Latency
In  Edge  Mode,  pose  detection  →  on-screen
visual/audio feedback must be ≤ 1 second 95% of
the time.
## 2.  Edge Processing
## Frame Rate
The Orin Nano must sustain ≥15 FPS while running
pose   estimation,   scoring   logic,   and   local   GUI
simultaneously.
-  Cloud Analysis Time For a 30-second MP4 video (≤50MB), Cloud Mode
must  return  the  complete  error  report  within  ≤2
minutes of upload completion.
## 4.  Edge Report
## Generation Time
In  Edge  Mode,  the  final  session  report  (scores,
GIFs,  feedback)  must  be  generated  within  ≤1
minute after the user clicks "End Session."
## 5.  Concurrent User
## Scalability
Cloud backend must process ≥5 simultaneous video
uploads without increasing average processing time
per video by more than 20%.
-  Setup Usability A  novice  user  must  be  able  to  unpack  hardware,
connect  cables,  launch  the  system,  and  complete
first-time calibration in ≤5 minutes without external
help.
-  Accessibility System must produce valid evaluations using only
the  OAK-D  camera  (Edge)  or  a  standard  1080p
camera (Cloud);  no  smartwatch,  bands,  or  sensors
required.
-  Security After   5   consecutive   failed   login   attempts,   the
account must be locked for exactly 30 minutes.
-  Data Encryption All  user  data  at  rest  (videos,  landmarks,  reports)
must be encrypted using AES-256 or equivalent.
-  Input Validation System  must  reject  any  uploaded  file  that  is  not
.mp4 or .mov or exceeds  MB, with a specific error
message stating the reason.

CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 45
## 11.  View Angle
## Validation
If  the  user's  torso  is  rotated  >30°  from  the  target
plane  (front/side),  the  system  must  detect  this  and
display an "Adjust Camera Angle" warning before
processing.
-  Low-Light Operation Pose  estimation  must  maintain  ≥90%  landmark
detection confidence in environments with ≥200 lux
ambient lighting.
## 13.  Data Persistence
## Reliability
100% of completed analysis sessions must be saved
and  retrievable  from  the  user  dashboard;  no  data
loss after a system restart.
## 14.  Rule Configuration
## Time
A developer must be able to activate or deactivate a
symbolic  rule  (e.g.,  “knee  valgus  check”)  for  an
exercise  in  ≤15  minutes  without  redeploying  the
entire system.

Table 11: Non-Functional Requirements


CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 46
4.2.3 Flowcharts of AI Module

Figure 13: AI Module Flowchart

































CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 47
## 4.2.4 Architecture Diagram


## Figure 14: Project Architecture Diagram
Figure 12 describes ExeVision’s architecture diagram. In Edge Mode, the user stands
in front of a Luxonis OAK-D camera that streams video directly to an NVIDIA Jetson
Orin  Nano  running  a  FastAPI  inference  service.  The FastAPI  service performs  real-
time  pose  estimation,  biomechanical  rule  checks,  and  Bi-LSTM  smoothness  scoring,
then pushes corrective cues to a local Next.js UI and stores session data in an on-device
SQLite database. When the session ends, the system generates a complete report within
one minute. In Cloud Mode, the user records a video on any smartphone, uploads it
through  the  Next.js  frontend  hosted  on  Vercel,  and  the  file  is  placed  in Supabase
Storage. A serverless function (Replicate) receives the public video link, runs the same
AI pipeline, and writes results back to a the Supabase database. The user then views the
analysis in the Web App.


CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 48
## 4.3 Intelligent Methods Used
## 4.3.1 Computer Vision & Feature Extraction
This component functions as the perceptual layer of the architecture, responsible
for  converting  raw  video  into  a  robust  structured  representation.  ExeVision
implements  a  multi-modal  feature  extraction  pipeline  that  captures  2D  or  3D
pose  landmarks  (depending  on  the  hardware  setup),  approximate  barbell
trajectory,  and  a  dynamic  reference  to  the  ground  plane.  Unlike  simple  pixel
analysis,  these  features  are  represented  as  skeletal  key  points  and  scene
references.   To   ensure   the   system   remains   robust   to   variations   in   user
anthropometry,  camera  distance,  and  perspective  distortion,  all  geometric
measurements are normalized by torso length and projected against a heuristic
floor plane defined by stable distal key points (e.g., ankles).
4.3.2 State-Aware Temporal Segmentation
Rather than relying on simple velocity thresholds which might fail if the user
pauses  while  doing  the  reps, this  module  utilizes  a  state  machine  to  track
landmark  kinematics  (e.g.,  hip  velocity).  This  approach  accurately  parses
movement  into  distinct  biomechanical  phases,  such  as  idle,  concentric,  and
eccentric. Crucially, this module incorporates a view-validation check prior to
full  analysis.  Once  validated,  the  system  triggers  the  relevant  downstream
symbolic microprograms.
## 4.3.3 Evaluation & Reasoning  Model
The   evaluation   stage   consists   of   two   complementary   AI   approaches.
First,  the  system  uses  Symbolic  Microprograms  (Rule-Based  AI) modular,
biomechanical  rules  that  activate  based  on  the  selected  exercise  and  camera
angle. Each rule outputs an error type, a severity percentile (0 to 100), and the
worst frame index.

Second,  the  system  incorporates  a  Neural  Smoothness  Scorer,  implemented
using  a  Bi-Directional  Long  Short-Term  Memory  (BiLSTM)  network.  This

CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 49
deep  learning  model  analyzes  sequences  of  normalized  distance  matrices  to
quantify movement smoothness, rhythm, and control. The neural score serves
as  an  advisory  signal  that  complements  but  does  not  override  the  rule-based
outputs.
## 4.3.4 Decision Feedback Generation
This  module  produces  the  final  assessment  output  delivered  to  the  user.  A  score
aggregation algorithm combines symbolic severity percentiles (primary signal) with the
BiLSTM  smoothness  score  (secondary  signal)  into  a  consolidated  quality  rating. For
the  feedback  generation,  the  system  uses  Dynamic  Template  Slot-Filling to  populate
predefined feedback templates with extracted error information such as type, severity,
and frame location. Additionally, a visual evidence retrieval algorithm maps detected
errors  to  their  corresponding  timestamps  and  extracts  short  three-frame GIFs  that
highlight the exact moment a deviation occurred. Together, these outputs form a clear,
interpretable   feedback   report   without   relying   on   hallucination-prone   generative
language models
## 4.4 Data Sources
The  project  will  use  primarily  3 sources of  data  source,  either  as  a  reference  for
extracting biomechanical rules or to train any neural networks such as Bi-LSTM or any
other models  deemed relevant. These sources are:
- FLEX Dataset
Largest fitness AQA dataset with 40+ hours of multimodal (RGB-D, sEMG)
multiview videos for 20 weight-loaded exercises (e.g., squats, deadlifts),
performed 10x by 38 subjects at novice/amateur/expert levels. Includes expert
annotations for keysteps, errors, AQA scores (0-100), and feedback via a
fitness knowledge graph; supports injury prevention and failure prediction
(Yin, H. et al, 2025)
- Fit3D Dataset
Contains over 3 million images from 611 recordings of 47 exercises across
multiple subjects, with 3D SMPLX meshes, skeletons, rep segmentations, and
Vicon mocap ground truth. Ideal for pose estimation, temporal segmentation,

CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 50
and multi-view analysis in gyms; 18GB training/1.4GB testing sets available
after registration (Fieraru, M. et al, 2021)
## • Custom Dataset
Author-captured videos, single-user, recordings of relevant exercises from
front & side views to check AI model consistency against the author’s own
judgement.
## 4.5 Technology Deployed
## 4.5.1 Hardware
## Attribute Description
Name NVIDIA    Jetson    Orin    Nano    Super
## Developer Kit
Operating System Linux (Ubuntu)
## GPU
NVIDIA Ampere architecture
1024 CUDA Cores 32 Tensor Cores
1020 MHz
CPU 6-Core ARM Cortex-A78AE v8.2 64-bit
## CPU
Memory 8GB 128-bit LPDDR5 102 GB/s

Table 12: NVIDIA Jetson Orin Nano Specifications
## Attribute Description
Name Luxonis OAK-D
Cameras 1×  12  MP  Sony  IMX378  color  (rolling
shutter)
2×  1  MP  OmniVision  OV9282  mono
(global shutter)
VPU Intel Movidius Myriad X
Depth Sensing Stereo baseline 7.5 cm, 1280×800 @ 120
fps
Connectivity USB 3.1 Gen 2 (10 Gbps) Type-C

Table 13: Luxonis OAK-D Specification

CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 51

## Attribute Description
Name HP Victus-15
## Operating System Windows 11
GPU NVIDIA Geforce RTX 3050
CPU 6-Core AMD   Ryzen   5   5600H   with
Radeon Graphics, 3.30 GHz
Memory 16 GB DDR4 RAM
## 512 GB SSD ROM
Table 14: HP Victus 15 Specifications
## 4.5.2 Software
## Attribute Description
Frontend Next.js, React, TypeScript
Backend Supabase, FastAPI
Database Supabase (PostgreSQL), SQLite
## Authentication Supabase
## Website Hosting Vercel
AI Model Server Replicate (cloud)
AI Model Language and Library Python, MediaPipe, OpenCV, Numpy (to
be added)
## Diagram Drawing Tools Draw.io, Canva
## Documentation Tools Microsoft Words
## Code Repository Hosting Github
## Table 15: Software Stack

CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 52
## 5 CONCLUSION
In conclusion, ExeVision aims to provide an efficient and comprehensive AI-powered
exercise  evaluation  solution  that  addresses  key  challenges  faced  by  gym-goers  and
individuals performing self-guided training.
In  Malaysia,  there  is  an  obvious  need  for  digital  transformation  in  the  fitness  and
exercise  domain  due  to  the  high  frequency  of  improper  exercise  execution,  limited
access  to  personal  trainers,  and  the  lack  of  structured,  actionable  feedback  for  users.
Throughout  this  report,  we  have  covered  in  depth  the  gaps  in  current  solutions  that
present  ExeVision  with  opportunities.  The  system  offers  unique  features  that  can
greatly  enhance  exercise  training  and  safety,  such  as  quantitative  form  scoring,
interpretable biomechanical feedback, and visual evidence of detected errors.
Along  with  outlining  a  detailed  project  scope  with  specified  module  integration  and
feature  specifications,  we  have  also  carried  out  a  thorough  analysis  of  system
requirements. We began by defining the project scope, creating a project management
strategy, and  designing  the  system  architecture,  data  model,  system  flow,  and  other
components after gaining a clearer understanding of the target users and their needs.
Next, the infrastructure and hosting environments will be set up, the system source code
will  be  developed,  and  the  system  design  will  be  implemented  using  the  specified
development  methodology.  To  ensure  that  both  the  functional  and  non-functional
requirements operate as intended, system testing will be conducted concurrently with
the implementation phase.







CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 53
## 6 SDG ALIGNMENT
This  project  directly  supports  Sustainable  Development  Goal  (SDG)  3:  Good  Health
and Well-Being, which seeks to ensure healthy lives and promote well-being for all at
all  ages  (United  Nations,  2015).  It  contributes  particularly  to  Target  3.4,  which
emphasizes  reducing  premature  mortality  from  non-communicable  diseases  (NCDs)
through prevention and health promotion.
According  to  the  World  Health  Organization  (WHO,  2023),  non-communicable
diseases such as cardiovascular disorders, diabetes, and obesity are responsible for more
than 70% of  global deaths. Regular, well-performed physical activity remains one of
the  most effective  measures  to  prevent  such  conditions.  However,  improper  exercise
techniques  and  poor  posture  can  lead  to  musculoskeletal  injuries,  discouraging  long-
term participation.
The  proposed  intelligent  fitness  evaluation  system  directly  supports  SDG  3.4  by
promoting safe, structured, and accessible physical activity through AI-driven posture
monitoring and form evaluation. Studies have shown that AI-based exercise guidance
and  motion  analysis  significantly  improve  movement  quality  and  reduce  injy  risks
(Gabrani et al., 2024; Díaz et al., 2024). By providing real-time or buffered feedback
on  posture  alignment  and  range  of  motion,  this  system  encourages  sustainable
engagement  in  physical  activity  and  helps  users  adopt  safer,  more  effective  training
habits.


CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 54
## 7 REFERENCES
Agresta, C., & Brown, A. (2015). Gait retraining for injured and healthy runners using
augmented feedback: A systematic literature review. Journal of Orthopaedic &
## Sports Physical Therapy, 45(8), 576–584.
https://doi.org/10.2519/jospt.2015.5823
Antón,  D.,  Kurillo,  G.,  Goñi,  A.,  Illarramendi,  A.,  &  Bajcsy,  R.  (2017).  Real-time
communication    for    Kinect-based    telerehabilitation.    Future    Generation
Computer Systems, 75, 72–81. https://doi.org/10.1016/j.future.2017.02.015
Centers for Disease Control and Prevention. (2025, February 5). Overcoming barriers
to physical activity. https://www.cdc.gov/physical-activity-basics/overcoming-
barriers/index.html
Chen, S., & Yang, R. R. (2020). Pose trainer: Correcting exercise posture using pose
estimation. arXiv. https://arxiv.org/abs/2006.11718
Collins, K. A., Huffman, K. M., Wolever, R. Q., Smith, P. J., Siegler, I. C., Ross, L.
M., Hauser, E. R., Jiang, R., Jakicic, J. M., Costa, P. T., & Kraus, W. E. (2022).
Determinants  of  dropout  from  and  variation  in  adherence  to  an  exercise
intervention:  The  STRRIDE  randomized  trials.  Translational  Journal  of  the
American College of Sports Medicine, 7(1), e000190.
https://doi.org/10.1249/TJX.0000000000000190
Díaz Jiménez, D., López Ruiz, J. L., González Lama, J., & Verdejo Espinosa, Á. (2024).
Assessing the sustainable alignment of a sensor-based connected health system
with SDGs: An evaluation model and case study. Smart and Sustainable Built
Environment. https://doi.org/10.1108/SASBE-01-2024-0012
Fieraru,  M.,  Zanfir,  M.,  Pirlea,  S.  C.,  Olaru,  V.,  &  Sminchisescu,  C.  (2021).  AIFit:
Automatic  3D  human-interpretable  feedback  models  for  fitness  training.  In
Proceedings  of  the  IEEE/CVF  Conference  on  Computer  Vision  and  Pattern
## Recognition (pp. 9919–9928).
Gabrani, G., Gupta, S., Verma, L., & Vyas, S. (2024). The role of health informatics in
achieving   sustainable   development   goals.   In   Sustainability   and   health

CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 55
informatics: A systems approach to address the  climate action induced global
challenge (pp. 73–98). Springer Nature Singapore.
Gentil, P., Soares, S., & Bottaro, M. (2017). Single vs. multi-joint resistance exercises:
Effects on muscle strength and hypertrophy. Asian Journal of Sports Medicine,
8(1), e40832. https://doi.org/10.5812/asjsm.40832
Gjestvang,  C.,  Stensrud,  T.,  &  Haakstad,  L.  A.  H.  (2019).  Are  changes  in  physical
fitness, body composition and weight associated with exercise attendance and
dropout    among    fitness    club    members?    BMJ    Open,    9(4),    e027987.
https://doi.org/10.1136/bmjopen-2018-027987
Gjestvang, C., Abrahamsen, F., Stensrud, T., & Haakstad, L. A. (2020). Motives and
barriers to initiation and sustained exercise adherence in a fitness club setting—
A one‐year follow‐up study. Scandinavian journal of medicine & science in
sports, 30(9), 1796-1805.
Gjestvang, C., Tangen, E. M., Arntzen, M. B., & Haakstad, L. A. H. (2023). How do
fitness  club  members  differentiate  in  background  characteristics,  exercise
motivation, and social support? Journal of Sports Science & Medicine, 22(2),
235–244. https://doi.org/10.52082/jssm.2023.235
Gymscore. (2025). Gymscore – AI fitness coach app for form analysis and personalized
feedback. https://www.gymscore.ai/ Gymscore
Hjaltason, M., & Gertrud, U. (2025). Learning to assess squat technique: Video-based
pose   analysis   with   classical   and   deep   learning   models   (Undergraduate
dissertation). Linnaeus University.
https://urn.kb.se/resolve?urn=urn:nbn:se:lnu:diva-139331
Ingledew, D. K., & Markland, D. (2008). The role of motives in exercise participation.
## Psychology & Health, 23(7), 807–828.
https://doi.org/10.1080/08870440701405704
Koh, Y. S., Asharani, P. V., Devi, F., Roystonn, K., Wang, P., Vaingankar, J. A., Abdin,
E.,  Sum,  C.  F.,  Lee,  E.  S.,  Müller-Riemenschneider,  F.,  Chong,  S.  A.,  &
Subramaniam,  M.  (2022).  Perceived  barriers  to  physical  activity  and  their

CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 56
associations  with  domain-specific  activity  and  sedentary  behaviour.  BMC
Public Health, 22, 1051. https://doi.org/10.1186/s12889-022-13431-2
Lu, Y., Leng, X., Yuan, H., Jin, C., Wang, Q., & Song, Z. (2024). Comparing the impact
of personal trainer guidance to exercising with others: Determining the optimal
approach. Heliyon, 10(2).
Luxonis. (2025). OAK-D product documentation.
https://docs.luxonis.com/hardware/products/OAK-D
MediaPipe. (2025). Pose landmarker. Google AI Edge.
https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
Mishra,  Y.,  Jaiswal,  A.,  Shukla,  A.,  Verma,  A.,  Verma,  H.,  &  Soni,  D.  (2025).  AI
human  fitness  tracker  using  computer  vision  with  MediaPipe.  International
Journal  for  Research  in  Applied  Science  and  Engineering  Technology,  13,
1659–1669. https://doi.org/10.22214/ijraset.2025.70547
Neilson, V., Ward, S., Hume, P., Lewis, G., & McDaid, A. (2019). Effects of augmented
feedback  on  training  jump  landing  tasks  for  ACL  injury  prevention.  Physical
Therapy in Sport, 39, 126–135. https://doi.org/10.1016/j.ptsp.2019.07.004
Ntoumanis,  N.,  Ng,  J.  Y.  Y.,  Prestwich,  A.,  Quested,  E.,  Hancox,  J.  E.,  Thøgersen-
Ntoumani, C., & Williams, G. C. (2021). A meta-analysis of self-determination
theory-informed  interventions.  Health  Psychology  Review,  15(2),  214–244.
https://doi.org/10.1080/17437199.2020.1718529
Nyman,  E.,  &  Armstrong,  C.  W.  (2015).  Real-time  feedback  during  drop  landing
training  improves  knee  kinematics.  Clinical  Biomechanics,  30(9),  988–994.
https://doi.org/10.1016/j.clinbiomech.2015.07.005
Parmar, P., & Tran Morris, B. (2017). Learning to score Olympic events. In Proceedings
of   the   IEEE   Conference   on   Computer   Vision   and   Pattern   Recognition
## Workshops (pp. 20–28).

CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 57
Parmar,  P.,  Gharat,  A.,  &  Rhodin,  H.  (2022).  Domain  knowledge-informed  self-
supervised   representations   for   workout   form   assessment.   In   European
Conference on Computer Vision (pp. 105–123). Springer Nature Switzerland.
Radhakrishnan,  M.,  Misra,  A.,  Balan,  R.  K.,  &  Lee,  Y.  (2020).  Gym  usage  behavior
and  desired  digital  interventions:  An  empirical  study.  In  Proceedings  of  the
ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (pp. 97–
107). https://doi.org/10.1145/3421937.3422023
Schoenfeld, B. J. (2010). The mechanisms of muscle hypertrophy and their application
to resistance training. Journal of Strength and Conditioning Research, 24(10),
2857–2872. https://doi.org/10.1519/JSC.0b013e3181e840f3
Storberget, M., Grødahl, L. H. J., Snodgrass, S., van Vliet, P., & Heneghan, N. (2017).
Verbal augmented feedback in lower extremity rehabilitation. BMJ Open Sport
&  Exercise  Medicine,  3(1),  e000256.  https://doi.org/10.1136/bmjsem-2017-
## 000256
Teixeira,  P.  J.,  Carraça,  E.  V.,  Markland,  D.,  Silva,  M.  N.,  &  Ryan,  R.  M.  (2012).
Exercise, physical activity, and self-determination theory. International Journal
of Behavioral Nutrition and Physical Activity, 9, 78.
https://doi.org/10.1186/1479-5868-9-78
United  Nations.  (2015).  Transforming  our  world:  The  2030  agenda  for  sustainable
development. https://sdgs.un.org/2030agenda
World    Health    Organization.    (2023,    November    2).    Climate    change    and
noncommunicable diseases: Connections.
Yin, H., Parmar, P., Xu, D., Zhang, Y., Zheng, T., & Fu, W. (2025). A decade of action
quality   assessment:   Trends,   challenges,   and   future   directions.   arXiv.
https://arxiv.org/abs/2502.02817
Yin,  H.,  Gu,  L.,  Parmar,  P.,  Xu,  L.,  Guo,  T.,  Fu,  W.,  &  Zheng,  T.  (2025).  FLEX:  A
large-scale   multimodal   multi-action   dataset   for   fitness   action   quality
assessment. arXiv. https://arxiv.org/abs/2506.03198

CAT405 System Requirements and Design Report Academic Session: 2025/2026

## 58
Zheng,  C.,  Wu,  W.,  Chen,  C.,  Yang,  T.,  Zhu,  S.,  Shen,  J.,  & Shah,  M.  (2023).  Deep
learning-based  human  pose  estimation:  A  survey.  ACM  Computing  Surveys,
## 56(1), 1–37.
Zhou, K., Cai, R., Wang, L., Shum, H. P., & Liang, X. (2024). A comprehensive survey
of     action     quality     assessment:     Methods     and     benchmarks.     arXiv.
https://arxiv.org/abs/2412.11149











## 8 APPENDIX
