---
layout: post
title:  "Comprehensive Q Learning"
date:   2020-05-03 18:00:00 +0700
img_thumbnail: /assets/img/thumbnail/Q-learning.png
img_header: /assets/img/header/Q-learning.png
description: "บทความนี้จะพูดถึง Q-Learning ตั้งแต่พื้นฐานที่สุดแบบ tabular Q-learning ไปจน Deep Q-Learning และรีวิว Extension ของมัน"
tags: ['reinforcement learning','deep learning']
---

#### List of Variable and Function
- $$s_t$$ = state ณ เวลา $$t$$
- $$a_t$$ = action ณ เวลา $$t$$
- $$r_t$$ = reward ที่ได้จากการทำ $$a_t$$ ใน $$s_t$$
- $$\gamma$$ = reward discount factor เอาไว้ควบคุมความสำคัญของ reward ในอนาคต
- $$a^\ast$$ = optimal action
- $$\pi$$ = Policy function เป็นฟังก์ชันที่จับคู่ระหว่าง state กับ optimal action หรือ $$a^{\ast}_t = \pi(s_t)$$
- $$\alpha$$ = learning rate
- $$\mathbb{E} [y \vert x=x_1]$$ คือค่าประมาณของ $$y$$ เมื่อค่าของ $$x$$ เท่ากับ $$x_1$$
- $$max_x f(x)$$ คือการหาค่า $$f(x)$$ ที่มากที่สุดโดยการปรับค่า x 
- $$argmax_x f(x)$$ คือการหาค่า x ที่ทำให้ได้ $$f(x)$$ มากที่สุด

<!-- # Markov Decision Process (MDP)
Markov Decision Process เป็นระบบแบบหนึ่งที่ต่อยอดมาจาก Markov Chain โดยที่มีการเพิ่ม action กับ reward เข้าไปให้
 -->
# Introduction to Q-learning
ถ้ายังจำกันได้ reinforcement learning คือการสร้าง policy ที่เป็นฟังก์ชันซึ่งจะแมพจาก state ไป action ที่จะทำให้เราได้ reward มากที่สุดในระยะยาว ถ้าเขียนเป็นรูปแบบทางคณิตศาสตร์ก็จะเป็นแบบนี้

\begin{equation}
\label{eq:eq_1}
   a^\ast_t = \pi(s_t)
\end{equation}

ซึ่งใน Q learning เนี่ย คือเราพยายามจะประมาณค่า reward ระยะยาวที่จะได้จากการทำแต่ละ action ในแต่ละ state ซึ่งจะแทนค่านั้นว่า $$Q(s,a)$$

\begin{equation}
\label{eq:eq_2}
   Q(s_t,a_t) = \mathbb{E}[r_t + \gamma^1r_{t+1}+ \gamma^2r_{t+2}+…|s = s_t, a= a_t]
\end{equation}


จะเห็นได้ว่า
- $$Q(s,a)$$ คือผลรวมของ reward ทั้งในเวลาปัจจุบัน และในอนาคต
- แต่ reward ในอนาคตจะถูกคูณด้วย $$\gamma$$ ยกกำลัง 1 2 3 4 5 6 … ไปเรื่อย ๆ
- ค่า $$\gamma$$ เป็น parameter ที่ชื่อว่า reward discount factor มีค่าระหว่าง 0–1 อยู่ที่ว่าเราให้ความสำคัญกับ reward ในอนาคตมากแค่ไหน ถ้าเราอยากให้ความสำคัญกับ reward ในอนาคตมากหน่อย เราก็ตั้ง $$\gamma$$ ให้เยอะ ๆ ไว้ เช่น 0.9999999

สมมติเรารู้ค่า $$Q(s,a)$$ ที่แม่นยำแล้ว เราก็สามารถคิด policy ของการเลือก action ในแต่ละ state ได้แบบง่ายๆเลย ก็คือในแต่ละ state ใด ๆ เราจะเลือก action ที่ให้ค่า $$Q(s_t,a_t)$$ มากที่สุด หรือเขียนได้ดังนี้

\begin{equation}
\label{eq:eq_3}
   a^\ast_t = \pi(s_t) = argmax_a (Q(s_t,a))
\end{equation}


## How to estimate $$Q(s_t,a_t)$$ value
ใน section ข้างบนเรารู้แล้วว่าค่า $$Q(s_t,a_t)$$ คืออะไร และเราสามารถนำไป optimize ค่า long-term reward ได้ยังไง เพราะฉะนั้นในส่วนนี้เราจะอธิบายวิธีการคำนวณค่า Q ที่แม่นยำเด้อ

จากสมการที่ \eqref{eq:eq_2} และ \eqref{eq:eq_3} ทำให้เราสามารถเขียนค่า $$Q(s_t,a_t)$$ ได้ในอีกรูปแบบหนึ่งโดยใช้หลักการง่าย ๆ คือ เราจะแทนค่า reward ในอนาคต ($$\gamma^1r_{t+1}+ \gamma^2r_{t+2}+…$$ ในสมการที่ \eqref{eq:eq_2}) ด้วย ค่า Q ใน state ถัดไป ($$\gamma Q(s_{t+1},a^\ast)$$) นั่นเอง ทำให้ได้สมการด้านล่าง ซึ่งเราเรียกมันว่า Bellman Equation

\begin{equation}
\label{eq:eq_4}
   Q(s_t,a_t) = r_t + \gamma max_a Q(s_{t+1},a) =r_t + \gamma Q(s_{t+1},a^\ast)
\end{equation}

จะเห็นได้ว่าในสมการด้านบนนั้น การที่เราจะประมาณค่า Q ได้ เราต้องมีตัวแปรอย่างน้อย 4 ตัว ก็คือ
- $$s_t$$ = state ปัจจุบัน
- $$a_t$$ = action ปัจจุบัน
- $$r_t$$ = reward ปัจจุบัน
- $$s_{t+1}$$ = state ในเวลาถัดไป

ในการฝึก robot ให้ประมาณค่า Q ได้นั้นเราจะให้ มันได้ลองผิดลองถูกใน environment ที่กำหนดให้ และในแต่ละ timestep นั้น มันจะเก็บ experience ที่ประกอบไปด้วยตัวแปร 4 ตัวด้านบนมาเพื่ออัพเดทค่า Q ของแต่ละ state และ action ไปเรื่อย ๆ จนกว่าค่า Q ของแต่ละ state และ action นั้นจะลู่เข้าสู่ค่าใดค่าหนึ่ง

![alt text](/assets/img/Q-learning/Q-learning-peudo.png)


ขั้นตอนโดยละเอียดมีดังนี้
1. สร้างตารางเอาไว้เก็บค่า $$Q(s,a)$$ ของทุก state-action โดยที่ตารางนั้นจะมีจำนวน row เท่ากับจำนวน state และจำนวน column เท่ากับจำนวน action 
2. สุ่มค่า Q มั่ว ๆ ใส่ในตารางไปก่อน หรือจะเริ่มด้วย 0 อะไรงี้ก็ได้
3. จากนั้นให้ robot เราได้โลดแล่นอยู่ใน environment โดยส่วนใหญ่ขั้นตอนนี้จะถูกแบ่งออกเป็น episode โดยที่ในแต่ละ episode จะมีขั้นตอนดังนี้
	- observe state ปัจจุบัน
	- เลือก action ด้วยการใช้ policy ในสมการที่ \eqref{eq:eq_2} และ epsilon-greedy 
		- Epsilon-Greedy เป็นการสำรวจ environment กล่าวคือเราจะเลือก action แบบสุ่มด้วยความน่าจะเป็น $$\epsilon$$ และเลือก action ที่ดีที่สุดตาม policy ด้วยความน่าจะเป็น $$1-\epsilon$$ 
		- โดยที่ค่า $$\epsilon$$ นั้นตอนแรกจะถูกตั้งไว้สูง ๆ ก่อน ก็คือประมาณ 1 และค่อย ๆ ลดลงมาเมื่อ train ไปเรื่อย ๆ
		- สาเหตุที่เราต้องใช้ epsilon-greedy ก็เพราะว่าเราต้องการให้ robot เราได้รู้จักกับ environment ให้ได้กว้าง ๆ เผื่อจะค้นพบ policy ที่ดีที่สุดจริง ๆ คือสมมติว่าเรามัวแต่เลือก action ที่เราคิดในตอนนี้ว่ามันดีที่สุดแล้ว เราก็จะไม่เคยได้ไปทดลองในทางอื่นเลย และจะไม่มีทางรู้ได้ว่า policy ที่เรามีอยู่ตอนนี้มันดีที่สุดจริง ๆ แล้วหรือยัง
	- ทำ action ที่เลือกในขั้นตอนที่แล้วใส่ environment และเก็บ reward และ state ถัดไป
	- อัพเดทค่า Q ด้วยสมการด้านล่าง 
			\begin{equation}
			\label{eq:eq_5}
			   Q(s_t,a_t) = Q(s_t,a_t) + \alpha (r_t + max_aQ(s_{t+1},a) - Q(s_t,a_t))
			\end{equation}
		- ซึ่งเป็นการคำนวณ error ระหว่างค่า target หรือเทอม $$r_t + max_aQ(s_{t+1},a)$$ และค่า $$Q(s_t,a_t)$$ ที่อยู่ในตาราง 
		- ค่า $$max_aQ(s_{t+1},a)$$ นั้นเราเลือกจากตารางได้เลย
		- จากนั้นนำค่า error มาอัพเดทค่า Q ที่อยู่ในตาราง ด้วยการคูณกับ learning rate หรือ $$\alpha$$ และบวกเข้าไปที่ค่า $$Q(s_t,a_t)$$ เดิม

	- ก๊อป $$s_{t+1}$$ มายัง $$s_{t}$$ เพื่อเป็นการก้าวสู่ step ถัดไป และวนไปเรื่อย ๆ จนกว่า episode จะสิ้นสุด ซึ่งถ้าเทียบกับเกมก็คือจนกว่าตัวละครจะตาย/ตกน้ำ/แพ้/ชนะ

## Implementation

ในบทความนี้จะยกตัวอย่างการสร้าง robot ให้เดินจากจุดหนึ่งไปอีกจุดหนึ่งได้โดยไม่ตกน้ำด้วยการใช้ Q learning นะครับ

ขั้นตอนโดยละเอียดมีดังนี้
1. สร้างตารางที่มีจำนวน row เท่ากับจำนวน state และจำนวน column เท่ากับจำนวน action
2. 




<!-- 
(สำหรับคนที่งงต้องอย่าลืมว่า $$Q(s_{t+1},a^\ast) = r_{t+1} + \gamma max_a Q(s_{t+2},a)$$ และมันก็จะเป็นแบบนี้ต่อไปเรื่อย ๆ )

![alt text](/assets/img/Q-learning/bellman-2.png) -->


