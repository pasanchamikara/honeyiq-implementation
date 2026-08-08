# HoneyIQ, Explained for a Curious Kid

## The Big Idea: A Trap House for Sneaky Visitors

Imagine you build a fake toy house. It looks real — it has fake toys, fake
doors, fake secrets inside. But it's not your *real* house. It's a trap.

If a sneaky burglar tries to break into it, they waste their time on the
fake house instead of your real one. And the whole time, you're secretly
watching them, writing down everything they do, and deciding what to do
about them.

That fake trap house is called a **honeypot**. It's a fake computer that
pretends to be interesting, so that hackers attack *it* instead of the
real computers.

HoneyIQ is the "brain" that watches the trap house and decides what to do
every time a visitor shows up: let them wander, quietly watch them, trick
them, slam the door, or call the guards.

## The Burglar's Seven Steps

Nobody breaks into a house in one move. A sneaky burglar goes through
stages, and HoneyIQ watches for exactly which stage a visitor is on:

1. **Reconnaissance** — peeking through the windows, checking if anyone's home
2. **Weaponization** — making their lockpick or tools
3. **Delivery** — sneaking the tool up to the door
4. **Exploitation** — actually picking the lock and getting in
5. **Installation** — setting up a hideout so they can sneak back in later
6. **Command & Control** — calling their friends to tell them "I'm in!"
7. **Actions on Objectives** — grabbing the toys and running off with them

The further down this list a visitor gets, the scarier they are. Someone
just peeking through a window (step 1) is way less alarming than someone
who's already inside stealing things (step 7).

## The Fortune-Teller Spinner

Here's a clever trick HoneyIQ uses: for every stage, it keeps a **spinner**
built from watching lots and lots of *past* burglars. The spinner doesn't
guess randomly — it's weighted based on what real sneaky visitors actually
did before.

So if a visitor is currently peeking through the window (stage 1), HoneyIQ
spins the spinner and it says something like: *"70% chance they just wander
off, 30% chance they go make a lockpick next."*

Add up all the spinner's chances of moving to a stage *scarier than where
they are right now*, and you get a single number called the
**escalation risk** — basically, "how likely is this visitor to get more
dangerous very soon?"

Different burglars have different personalities too — HoneyIQ calls these
**intents**:

- **Stealthy** — slow, quiet, patient, trying not to be noticed
- **Aggressive** — fast and loud, doesn't care about getting caught
- **Targeted** — focused, going straight for one specific thing
- **Opportunistic** — grabs whatever's easy, not picky

The spinner is different for each personality, because a stealthy burglar
and an aggressive one don't behave the same way.

## Red, Yellow, Green Light

A single escalation-risk number is hard to think about, so HoneyIQ turns it
into a simple traffic light:

| Escalation risk | Light  |
|---|---|
| below 35% | 🟢 Low |
| 35% – 65% | 🟡 Medium |
| above 65% | 🔴 High |

## The Cheat Sheet

Now for the cleverest part. HoneyIQ has a **cheat sheet** — a simple table
with 7 rows (the burglar's stage) and 3 columns (the traffic light color).
You look up the row and column, and it tells you exactly what to do. No
guessing, no complicated math in the moment — just look it up, like a
recipe card:

| Stage ↓ / Risk → | 🟢 Low | 🟡 Medium | 🔴 High |
|---|---|---|---|
| Reconnaissance | Allow | Log | Log |
| Weaponization | Log | Log | Troll |
| Delivery | Log | Troll | Troll |
| Exploitation | Troll | Block | Block |
| Installation | Block | Block | Alert |
| Command & Control | Block | Alert | Alert |
| Actions on Objectives | Alert | Alert | Alert |

Notice the pattern: the further down and further right you go, the
scarier the response gets. That's on purpose — early, low-risk visitors get
a gentle response, and late-stage, high-risk visitors get the strongest one.

## The Five Things the Trap House Can Do

Those words in the cheat sheet are the five possible responses, from
gentlest to strongest:

1. **Allow** — let them wander through, untouched
2. **Log** — quietly write down everything they do, keep watching
3. **Troll** — give them fake treasure and waste their time (a "tarpit")
4. **Block** — slam the door and cut off the connection
5. **Alert** — ring the alarm bell and call the guards immediately

## Three Special Exception Rules

After looking up the cheat sheet, HoneyIQ checks three special exceptions
before doing anything:

- **Rule 1 — Just a normal visitor.** If it turns out this isn't a burglar
  at all, just an ordinary guest, always just **Allow** them. Don't punish
  innocent people no matter what the cheat sheet said.
- **Rule 2 — A super-spreading danger.** Some "burglars" are less like a
  person and more like a wildfire or a nasty cold that spreads to everyone
  it touches. If HoneyIQ spots one of those, it bumps the response up one
  level tougher than the cheat sheet said — better safe than sorry.
- **Rule 3 — Knocking way too fast.** If a visitor has been attacking over
  and over and over very recently, HoneyIQ also bumps the response up one
  level, even if the cheat sheet alone wouldn't have.

"Bumping up one level" just means sliding one step to the right along the
list of five responses — Log becomes Troll, Troll becomes Block, and so on
(Alert is already the top, so it just stays Alert).

## The Worry-o-Meter (a bonus, just for the grown-ups watching)

Alongside the actual decision, HoneyIQ also writes down one more number
for the humans watching the dashboard: a **"worry score"** from 0 to 1. It
mixes together how late-stage the visitor is, how likely they are to get
scarier, how dangerous their attack type is, and how often they've been
knocking lately. This number doesn't change what the trap house *does* —
it's purely there so a human glancing at a chart can see "oh, that one was
a 0.9, that's a scary one" at a glance.

## The Guessing Guard

Before HoneyIQ can even use the cheat sheet, something has to figure out
*what kind* of knock this even is — is it a nosy peek, or an actual
break-in attempt? That job belongs to a separate little helper called the
**classifier**. It looks at clues about the connection (how big it is, how
long it lasted, how it behaves) and makes its best guess about what type of
attack it might be — a bit like a guard peering through a peephole and
guessing "mailman or burglar?" based on how they're dressed and how they
knock.

This guesser *does* learn from examples — it studies lots of
example knocks ahead of time so it gets better at guessing. But it's a
fairly simple kind of guesser (called a "random forest," basically a big
pile of yes/no questions voting together), not the giant, deep-learning
kind of AI "brain" you might hear about elsewhere. It's small and honest
about its limits, and its only job is guessing the *type* of visitor — the
cheat sheet and rules handle everything else.

## Why Not Just Use a Big Fancy AI Brain?

An earlier version of HoneyIQ actually did try to use a much bigger,
fancier learning "brain" (something called a DQN, a kind of deep-learning
system) to decide what to do. It worked, but nobody — not even the people
who built it — could fully explain *why* it made any particular decision.
It was a mysterious black box.

HoneyIQ was rebuilt around the cheat sheet instead, on purpose, because a
cheat sheet has a superpower a mystery-brain doesn't: **you can always ask
"why did you do that?" and get a real answer.** "Because the visitor was on
step 5, the spinner said high risk, and Rule 3 bumped it up one more
level" is something a person can check, argue with, and trust. That's much
more important for guarding a house than being fancy.

## Putting It All Together

Every time a visitor knocks on the trap house's door, here's the whole
story, start to finish:

1. The **guessing guard** looks at the knock and guesses what type of
   visitor this might be.
2. HoneyIQ checks which of the **seven burglar steps** they seem to be on.
3. The **fortune-teller spinner** (tuned to this visitor's personality)
   estimates how likely they are to get scarier soon — the escalation risk.
4. That risk becomes a **red, yellow, or green light**.
5. HoneyIQ looks up the stage and the light on its **cheat sheet** to get a
   starting response.
6. It checks the **three exception rules** in case the response needs to be
   gentler (normal visitor) or tougher (super-spreader, or knocking too fast).
7. It carries out one of the **five responses** — Allow, Log, Troll, Block,
   or Alert — and writes a **worry score** in its notebook for the humans.

No mystery, no magic, no giant AI brain required — just a clear cheat
sheet, three fair exception rules, and a spinner built from watching lots
of past burglars.
