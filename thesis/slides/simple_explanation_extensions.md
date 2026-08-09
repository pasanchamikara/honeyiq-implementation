# HoneyIQ Extensions — Plain-Language Explanation

Companion to [`simple_explanation.md`](simple_explanation.md). Read that
one first if you haven't — this covers what was added on top of it.

## The three gaps

The original HoneyIQ could tell *what kind* of attacker was knocking and
*what to do* about it. Three things were still missing:

1. **Every simulated attacker sounded the same.** The fake traffic used
   to test the system was regenerated from scratch, independently, every
   single moment — so a "big, noisy" attack session and a "small, quiet"
   one were pure luck, not something the system deliberately varied. And
   every innocent visitor looked exactly like every other innocent
   visitor.
2. **The "how worried should I be" gauge was crude.** It only asked "was
   the last 20 knocks an attack, yes or no?" — treating a nosy peek and a
   full break-in the same way, and forgetting instantly the moment a
   knock fell off the back of that 20-knock list.
3. **Nothing ever changed on its own.** The rulebook and its cutoff
   numbers were carved in stone. If the same troublemaker came back a
   week later, the system had no memory of them at all.

## Fix #1: Give each visitor a consistent "character"

Instead of re-rolling dice for every single feature of every single knock,
each simulated visitor now gets one "size" dial set at the start of their
visit — some sessions are consistently big and loud, others consistently
small and quiet, for the whole visit. And instead of one boring flavor of
"innocent visitor," there are now three: an ordinary person, a busy web
crawler, and a quick monitoring check-in — each with its own realistic
pattern. The guessing guard (the classifier from the original
explanation) was retrained on this richer, more varied practice data, and
still guesses correctly 99.4% of the time.

## Fix #2: A gauge that fades instead of snapping

The old "how worried" gauge was a strict 20-knock counter — 21 knocks ago
mattered exactly as much as 1 knock ago until it vanished off the list.
The new gauge instead fades smoothly, like a dimmer switch, and it also
cares about *how scary* each knock was, not just whether it happened.
Both gauges are always running side by side; the old snap-counter is
still what's actually used by default, since the fade-gauge needs its own
careful checking before it's trusted for real decisions.

## Fix #3: Remembering repeat troublemakers, and a self-tuning dial

**Reputation.** The trap house now keeps a "trouble score" for each
visitor that quietly fades over time (halving roughly every 6 hours) but
never fully disappears the moment they leave. If that score gets high
enough, the visitor gets treated more harshly **even if this particular
knock looks perfectly innocent** — like a shop that keeps an eye on a
known shoplifter even when they're just browsing. This is deliberately
strict: it means someone who was briefly a problem stays watched for a
while even after they've calmed down, which is a fair trade for not
forgetting repeat troublemakers instantly.

**A self-tuning dial — but an honest one.** One of the three exception
rules (the one about "knocking too fast") can now nudge its own trigger
point up or down to avoid crying wolf too often or staying silent too
long. Here's the honest part: this dial does **not** claim to make better
*decisions* — nobody has told the system "that alarm was right" or "that
alarm was wrong," so it has no way to know. All it can honestly do is
watch its own alarm frequency and keep it from getting annoying. That's a
real, useful thing to build. Claiming more than that would be dishonest.

## Should the trap house get an AI brain after all?

This project already tried that once — a fancier "big AI brain"
(mentioned as an aside in the original explanation) was actually built
and tested. It didn't go well: the brain learned a cheap trick — treat
almost everything as dangerous — instead of actually learning to tell
visitors apart, because its practice sessions were almost entirely
break-in attempts. It also only ever practiced against one type of
visitor personality, so it wouldn't have handled the others well.

On top of that, the tool used to *measure* how well the system was doing
had its own embarrassing mistake — twice — where it accidentally graded
a decision against the wrong moment in time, making the results look far
worse than reality until someone caught it. That's a warning sign: if the
grading tool can get confused that easily, a "big AI brain" being trained
on very similar grading logic could get confidently, invisibly, wrong in
the same way — and you might never notice, because a wrong brain doesn't
raise its hand and say "I might be broken."

So the honest answer is: **not yet, and not like that.** The repeat-
troublemaker memory and the honest self-tuning dial do the job of
"getting smarter over time" without needing a mysterious brain, a long
practice period, or blind trust in a grading system that has already
been wrong twice. If a learning brain is ever added, the plan is to keep
it tiny (choosing between a handful of pre-approved settings, not
inventing new rules from scratch) and to only let it learn from a real
human saying "yes, that alarm was correct" or "no, that one was a
mistake" — a feedback path that doesn't exist yet and would need to be
built first, honestly, before any brain gets to learn from it.

## Bottom line

Same trap house, same one-page rulebook, same "you can always ask why."
Now it also: sounds more like a real neighborhood instead of a
simulation, has a gauge that fades instead of snapping, remembers repeat
troublemakers fairly, and keeps one of its alarms from crying wolf — all
without adding a single line of code a human can't read and check by
hand.
