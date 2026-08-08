# HoneyIQ — Plain-Language Explanation

## What problem does this solve?

Imagine a bank. The bank knows criminals exist and try to break in. So they set up a fake vault that looks real — stocked with fake money — right next to the real one. If a criminal goes for it, the bank knows immediately it is being targeted, and can watch what the criminal does without risking anything real. That fake vault is a **honeypot**.

The challenge is: what should the bank *do* once it notices someone poking around the fake vault? Should it:
- Stay quiet and keep watching? (learn more about the attacker)
- Sound the alarm? (stop the threat now)
- Feed the criminal false information to waste their time?
- Lock them out?

Getting this decision wrong either tips off the attacker (they run away before you learn anything) or lets them cause real damage.

**HoneyIQ answers that question automatically.**

---

## What is a Cyber Kill Chain?

Hackers rarely launch full attacks from the first click. They follow steps — almost like a recipe:

| Step | What the hacker does | Example |
|------|---------------------|---------|
| 1. Reconnaissance | Look around, map the network | Port scanning |
| 2. Weaponization | Build the attack tool | Crafting malware |
| 3. Delivery | Send the weapon in | Phishing email |
| 4. Exploitation | Break into a system | Running the exploit |
| 5. Installation | Leave a back door | Installing a rootkit |
| 6. Command & Control | Talk back to their server | C2 beacon |
| 7. Actions on Objectives | Do the damage | Steal data, destroy files |

This 7-step model is called the **Cyber Kill Chain** (Lockheed Martin, 2011). Knowing *which step* the attacker is on tells you how serious the threat is and what to do about it.

---

## What does HoneyIQ do?

HoneyIQ is a **smart honeypot manager**. It watches the fake system, figures out how dangerous the situation is, and picks the best response — automatically, and in a way a human can understand and check.

### The two key pieces

**1. The Attacker Model**
HoneyIQ simulates four types of attacker so the system can be tested thoroughly:

| Attacker Type | Behaviour |
|---------------|-----------|
| Stealthy | Slow, patient, tries not to be noticed |
| Aggressive | Fast, noisy, goes straight for damage |
| Targeted | Focused on one specific goal |
| Opportunistic | Random, tries many things at once |

Each attacker follows the Kill Chain using probability mathematics (Markov chains) so their behaviour is realistic and varied.

**2. The Decision Policy (SEDM)**
The Stage-Escalation Decision Matrix is a simple lookup table with two inputs:
- **Where is the attacker in the Kill Chain?** (steps 1–7)
- **How likely are they to jump to a more dangerous step?** (Low / Medium / High)

From those two answers, it outputs one of five responses:

| Response | Meaning |
|----------|---------|
| ALLOW | Normal traffic, do nothing |
| LOG | Suspicious — record it quietly |
| TROLL | Feed fake information, waste their time |
| BLOCK | Cut the connection |
| ALERT | Sound the alarm, call a human |

The entire policy fits in a single table you can print on one page. There is no "black box" — every decision can be explained in plain English.

---

## How does it know when to escalate?

HoneyIQ computes a **composite threat score** (0 to 1) from four signals:

- **45%** — How severe is the current attack type?
- **35%** — How far along the Kill Chain is the attacker?
- **15%** — How fast are they likely to escalate?
- **5%** — How many attacks have happened so far in this session?

Score below 0.15 = Benign. Score above 0.75 = Critical. The score drives both the response decision and the automatic alarm.

---

## What were the results?

Tested across 120 simulated attack sessions (30 per attacker type, 200 steps each):

| Attacker | Threats Detected | False Alarms |
|----------|-----------------|--------------|
| Stealthy | **99.09%** | 35.56% |
| Aggressive | **99.47%** | 6.67% |
| Targeted | **99.48%** | 3.33% |
| Opportunistic | **99.41%** | 15.00% |

Compared to a Deep Q-Network (an AI that has to be trained for thousands of rounds), the SEDM:
- Detects threats just as well (>99%)
- Requires **zero training time**
- Can be **read and audited by a human**
- Works equally well against all four attacker types without any retuning

---

## Why does it matter?

Most AI-based security tools are "black boxes" — they give you an answer but cannot explain *why*. In a real security operations centre, analysts must justify every decision to managers, auditors, and sometimes courts. HoneyIQ shows that a fully transparent, rule-based policy can match the detection performance of a neural network, while remaining completely explainable.

**Bottom line:** HoneyIQ is a honeypot management system that is smart enough to catch nearly every attack, honest enough to explain every decision, and simple enough to adjust without a data scientist.
