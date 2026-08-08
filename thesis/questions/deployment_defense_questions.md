# Preparing to Defend the Deployment Design

This note works through the questions an examiner is likely to raise about
taking HoneyIQ off the page and onto a real cloud instance, fronted by a
proper TLS certificate from Let's Encrypt via certbot. It is written as a
single argument rather than a checklist, because the questions below are
not independent: how you answer the first one (what the deployment
actually demonstrates) constrains what you can honestly say about all the
others. The supporting literature for each section lives in
[`literature_references.md`](literature_references.md) and
[`../questions-new/literature_references_new.md`](../questions-new/literature_references_new.md);
the architecture referred to throughout is sketched in
[`../latex/figures-new/arch_oracle_deployment_new.png`](../latex/figures-new/arch_oracle_deployment_new.png)
and [`arch_certbot_tls_flow_new.png`](../latex/figures-new/arch_certbot_tls_flow_new.png).

## Start with what the deployment is actually for

Before anything else, decide honestly whether this deployment is part of
the thesis's evaluated contribution or a demonstration of engineering
feasibility sitting alongside it. Chapters 4 to 6 evaluate SEDM and DQN
entirely inside the simulation; if the live deployment doesn't produce
numbers that appear in those chapters, say so plainly. Calling it a
feasibility demo rather than a new experiment closes off a whole line of
questioning about why the live results don't match the paper's headline
figures, and it is the honest framing anyway: the deployment shows that
the policy can run against real infrastructure, not that it performs the
same way there.

That framing then answers the first hard question almost by itself. The
99%+ detection and 0% false-positive results in the thesis come from
synthetic, cleanly-separated UNSW-NB15-style features. The discussion
chapter is already explicit that this is a property of the simulator, not
a claim about real traffic (§5.4), and the feature-noise sweep backs that
up directly: accuracy falls from 99.85% to 87.65% and the false-positive
rate rises from 0% to 10% once the input signal is made realistically
noisy. A live honeypot is exactly the setting where that noise shows up
for real, so the deployment should be pitched as the next step in testing
that degradation, not as a repeat of the clean-data result.

## Why this infrastructure, and what it's isolated from

An examiner may reasonably ask why Oracle Cloud rather than AWS, GCP, or
Azure. The honest answer should be about the actual constraints that
drove the choice: budget (free-tier compute suits a student project),
and the fact that the instances were already provisioned. What matters
more than the provider is being able to describe the isolation clearly:
which network the honeypot instance sits in, which security rules limit
what's reachable from the internet, and what would happen if the honeypot
were compromised and used as a pivot point into anything else running
under the same account. The architecture sketch treats this as two
separate machines, one exposed and one not, precisely so that question has
a concrete answer rather than a hand-wave.

## What a real certificate does to a fake service

The TLS question cuts both ways and it's worth deciding, in advance,
which side of the argument to take. A trusted certificate makes the
honeypot indistinguishable from a genuine HTTPS service to automated
scanners, which is the whole point of running OpenCanary in the first
place: better fidelity means better data on real attacker behaviour. But
there's a second-order effect worth knowing about, because an examiner
who has read the honeypot-fingerprinting literature might raise it first:
static or default TLS certificates and banner metadata are a documented
way researchers and attackers alike fingerprint honeypots at scale
(Vetterl & Clayton's fingerprinting framework is the standard reference
here). A certbot-issued certificate, renewed automatically like a real
service's would be, removes that particular tell, which is an argument
for doing it this way rather than just a nice-to-have.

The practical complication is that certbot's usual HTTP-01 challenge
needs port 80 reachable from the internet, which puts the certificate
issuance path itself inside the same attack surface the honeypot is meant
to observe. The cleaner answer is to use a DNS-01 challenge instead, which
proves domain control through a DNS TXT record rather than an inbound
connection, so the honeypot instance never needs port 80 open for
certificate purposes at all. That's the flow sketched in
`arch_certbot_tls_flow_new.png`. It's also worth having a one-line answer
ready for what happens if renewal silently fails mid-experiment: the
answer should be that renewal failures are monitored and alerted on, not
discovered when the data collection pipeline goes quiet.

## Does the interpretability argument survive contact with reality

The thesis leans heavily on SEDM's auditability: any action can be traced
back to a specific matrix cell and override rule (§5.3). That property
doesn't disappear once the system is running on real infrastructure,
because the decision function is still deterministic given whatever state
it's handed. What does change is the quality of that state: real
deployment introduces observation noise from classifier misclassification
and from delayed or dropped events coming through the OpenCanary
integration layer, conditions the oracle-versus-classifier comparison in
Chapter 4 never has to deal with. Framing the deployment as a way to test
auditability under that noise, rather than as a rerun of Chapter 4's
numbers, is the more defensible position and also the more interesting
one.

The same section of the discussion chapter already flags SEDM's
determinism as a double-edged property: predictable and easy to audit,
but also fingerprintable by an adversary who can learn the policy and
craft states that provoke a weaker response. That's not a hypothetical
worry once the honeypot is internet-facing rather than simulated. The
recent literature on adaptive, game-theoretic honeypot allocation (see the
Bayesian Stackelberg and moving-target-defense references in the
literature file) is directly about this scenario, and it's worth having
read enough of it to say something more specific than "future work" if
pressed on whether a fixed matrix policy is safe to expose to real
adversaries.

## The parts that aren't really technical questions

Two more questions are worth preparing for even though they aren't about
architecture. First, whether running an internet-facing honeypot that
captures IPs and payloads from real, uninvited traffic falls under the
institution's ethics approval, and what happens to that data afterwards.
This gets asked in security-adjacent vivas regardless of how solid the
engineering is, so it's worth having an actual answer about retention and
anonymisation rather than improvising one in the room. Second, the blast
radius question from earlier isn't just a network diagram exercise: if
something goes wrong, who is accountable for what the honeypot was used
for while compromised. Neither of these needs the diagrams or the
literature review to answer well; they need a straight, thought-through
sentence each.
