---
layout: post
title: "Flexible BERT: Teaching a Model to Spend the Right Amount of Thought"
date: 2026-07-02 01:02:38 +0530
categories: bert efficiency dynamic-inference
mathjax: true
excerpt: "A teaching note on Flexible BERT: width routing, early exits, neural grafting, and how adaptive inference lets a model spend compute only where it helps."
---

# Content

1. TOC
{:toc}

---

# The Core Problem

Imagine a professor grading a stack of exam answers.

Some answers are obvious. One sentence is enough to know the student understood the idea.
Some answers are ambiguous. The professor must read carefully, compare details, and think longer.

Now imagine a rule that says:

> Spend exactly the same amount of time on every answer.

That is wasteful. Easy cases are over-processed. Hard cases may still need all the attention they can get.

Standard BERT behaves like that fixed-rule professor. Every input goes through the same full stack:

```text
Input -> Layer 1 -> Layer 2 -> ... -> Layer 12 -> Prediction
```

The paper **A Flexible BERT Model Enabling Width- and Depth-Dynamic Inference** asks a better question:

> Can BERT learn how much computation each input deserves?

The answer is yes. Flexible BERT makes computation adaptive along two axes:

1. **Width**: how much of each Transformer layer should be active?
2. **Depth**: how many layers should the input pass through before prediction?

The mental model is triage. Easy inputs get a quick checkup. Hard inputs get the full diagnostic path.

<figure class="flexi-figure">
  <img src="/assets/images/flexibert_mental_model.png" alt="Mental model of Flexible BERT dynamic inference">
  <figcaption>A useful high-level mental model: each input gets routed through just enough BERT.</figcaption>
</figure>

# A Fixed Model Versus An Adaptive Model

Standard BERT gives every input the same computational budget. Flexible BERT gives every input a route.

<div class="flexi-diagram flexi-two-col">
  <section>
    <h3>Standard BERT</h3>
    <div class="flexi-stack">
      <span>Input</span>
      <span>Layer 1: 100%</span>
      <span>Layer 2: 100%</span>
      <span>Layer 3: 100%</span>
      <span>...</span>
      <span>Layer 12: 100%</span>
      <span>Prediction</span>
    </div>
    <p>Same width. Same depth. Same cost.</p>
  </section>
  <section>
    <h3>Flexible BERT</h3>
    <div class="flexi-stack">
      <span>Input</span>
      <span>Gate: choose 50%</span>
      <span>Layer 1</span>
      <span>Gate: choose 25%</span>
      <span>Layer 2</span>
      <span>Off-ramp: confident?</span>
      <span>Exit or continue</span>
    </div>
    <p>Different inputs can use different paths.</p>
  </section>
</div>

That is the conceptual shift. The model is no longer just a fixed function. It becomes a small decision system wrapped around a Transformer.

# What Does Width Mean?

A BERT layer is not one indivisible block. It contains attention heads and feed-forward network neurons. Width controls how many of those components are active.

Flexible BERT uses width ratios such as:

```text
25%, 50%, 75%, 100%
```

At each layer, a small gating module looks at the current `[CLS]` representation and predicts which width ratio to use next.

<div class="flexi-diagram">
  <h3>Width Is Capacity Inside A Layer</h3>
  <div class="width-grid" aria-label="Width ratios">
    <div>
      <strong>25%</strong>
      <span class="width-bar w25"></span>
      <p>Use a small slice.</p>
    </div>
    <div>
      <strong>50%</strong>
      <span class="width-bar w50"></span>
      <p>Use half the layer.</p>
    </div>
    <div>
      <strong>75%</strong>
      <span class="width-bar w75"></span>
      <p>Use most components.</p>
    </div>
    <div>
      <strong>100%</strong>
      <span class="width-bar w100"></span>
      <p>Use the full layer.</p>
    </div>
  </div>
</div>

A single input may take a route like this:

```text
Layer 1  -> 50% width
Layer 2  -> 25% width
Layer 3  -> 75% width
Layer 4  -> 50% width
...
```

This matters because Flexible BERT does not choose one small model for all future inputs. It chooses width **per input, per layer**.

# How Does The Gate Learn?

This is one of the most elegant parts of the paper.

The gate does not need a human to label examples as easy or hard. The model generates its own supervision by asking:

> What is the smallest subnet that gets this example right?

For a training example, run the available width subnets:

<div class="flexi-diagram">
  <h3>Gate Label From The Smallest Correct Subnet</h3>
  <div class="gate-labels">
    <span class="bad">25% wrong</span>
    <span class="bad">50% wrong</span>
    <span class="good">75% correct</span>
    <span class="good">100% correct</span>
  </div>
  <p class="flexi-callout">Gate target: choose 75%.</p>
</div>

If 25% already gets the answer right, the gate learns that the input is easy. If only 100% works, the gate learns that the input is hard.

This is a powerful teaching pattern: instead of manually defining input difficulty, let the model discover difficulty through which subnet succeeds.

# What Does Depth Mean?

Depth is how many Transformer layers the input passes through.

Standard BERT-Base has 12 layers and always uses all of them. Flexible BERT adds intermediate classifiers called **off-ramps**. In the paper, off-ramps are placed after layer 6 and layer 9, with the final classifier after layer 12.

<div class="flexi-diagram">
  <h3>Depth Is How Long The Input Keeps Thinking</h3>
  <div class="depth-road">
    <span>Input</span>
    <span>Layers 1-6</span>
    <span class="ramp">Exit?</span>
    <span>Layers 7-9</span>
    <span class="ramp">Exit?</span>
    <span>Layers 10-12</span>
    <span>Final</span>
  </div>
</div>

At each off-ramp, the model asks:

> Am I confident enough to stop?

If yes, prediction happens early. If no, the input continues to deeper layers.

# Entropy: The Confidence Meter

For classification tasks, the paper uses entropy as an uncertainty score.

Low entropy means the probability distribution is sharp:

```text
Positive: 0.998
Negative: 0.002
```

That is a confident prediction. The input can exit early.

High entropy means the distribution is flat:

```text
Positive: 0.52
Negative: 0.48
```

That is uncertain. The input should continue.

The rule is simple:

```text
low entropy  -> confident -> exit
high entropy -> uncertain -> continue
```

For regression tasks such as STS-B, there is no softmax distribution. The paper instead uses prediction stability between layers: if the prediction is no longer changing much, the input can exit.

# Neural Grafting: Do Not Throw Away Weak Parts

Flexible BERT is not only about routing at inference time. It also has to train many subnetworks inside one shared model.

That creates a problem. Small subnetworks use only a subset of attention heads and FFN neurons. Large subnetworks use more. Some components can become under-trained or inactive.

A pruning mindset would say:

> If a component is weak, remove it.

Neural grafting says:

> If a component is weak, rehabilitate it.

The model ranks attention heads and FFN neurons by importance. Then, inside a layer, weaker components receive a small weighted signal from stronger components.

<div class="flexi-diagram">
  <h3>Neural Grafting As Rehabilitation</h3>
  <div class="graft-row">
    <div class="head strong">Strong head</div>
    <div class="graft-arrow">alpha signal</div>
    <div class="head weak">Weak head</div>
    <div class="graft-result">Updated weak head</div>
  </div>
  <code>weak_new = alpha * strong + (1 - alpha) * weak</code>
  <p>In the paper, alpha = 0.1 worked well across tasks.</p>
</div>

This happens layer by layer. The same idea is applied to attention heads and FFN neurons.

The important intuition is that grafting is not replacement. The weak component keeps most of its identity and receives a small nudge from a stronger peer.

# The Full Training Story

Flexible BERT is built in stages.

<div class="flexi-diagram">
  <h3>Training Pipeline</h3>
  <ol class="pipeline">
    <li><strong>Train width subnets:</strong> 25%, 50%, 75%, and 100% width learn from a full BERT teacher.</li>
    <li><strong>Apply neural grafting:</strong> weak heads and FFN neurons are nudged using stronger components.</li>
    <li><strong>Train depth subnets:</strong> depth ratios 0.5, 0.75, and 1.0 correspond roughly to 6, 9, and 12 layers.</li>
    <li><strong>Fine-tune:</strong> all subnetworks are tuned on the downstream task.</li>
    <li><strong>Train gates and off-ramps:</strong> the model learns width choices and early-exit decisions.</li>
  </ol>
</div>

Under the hood, the subnet training uses knowledge distillation. The smaller paths learn from the behavior of larger or full teacher models.

# What The Paper Reports

The headline result is not merely "smaller BERT is faster."

The stronger claim is:

> Adaptive BERT can improve the performance-cost tradeoff because it spends compute where it is useful.

Compared with BERT-Base, the paper reports:

- about **45% fewer computations**
- about **2.3x CPU speedup**
- **+1.3 average GLUE points**
- **+2.2 SQuAD F1 points**

The exact numbers depend on task and budget, but the direction is clear: dynamic routing gives a better Pareto frontier than simply picking one fixed compressed model.

# Why GPU Speedup Is Harder Than FLOP Reduction

One subtle point in the paper is worth remembering.

Width reduction lowers FLOPs, but that does not always translate into proportional GPU latency gains. GPUs are good at dense parallel computation, and irregularly using parts of a layer can be harder to exploit.

Depth reduction is more directly useful for latency because skipping layers removes entire blocks of work.

So a practical serving lesson is:

```text
width routing -> good for reducing arithmetic
depth routing -> often more visible for latency
width + depth -> best conceptual tradeoff
```

# How This Differs From Related Ideas

Flexible BERT sits at the intersection of several model-efficiency ideas.

| Idea | Core question | Flexible BERT difference |
|---|---|---|
| Distillation | Can one smaller model imitate a larger teacher? | Flexible BERT trains many usable paths inside one model. |
| Pruning | Which weak components can we delete? | Neural grafting strengthens weak components instead. |
| Early exit | Can easy inputs stop at an intermediate layer? | Flexible BERT combines early exit with width routing. |
| DynaBERT | Can one model support multiple width/depth subnets? | Flexible BERT chooses paths dynamically for each input. |
| MatFormer | Can submodels be nested inside a larger model? | Flexible BERT focuses on runtime routing, not only static submodel selection. |
| Mixture of Experts | Which expert modules should handle this input? | Flexible BERT chooses how much of each BERT layer to use. |

# A Professor's Mental Model

If I were teaching this in a classroom, I would ask students to remember one image:

> Flexible BERT is a hospital triage desk attached to BERT.

The patient is the input.

The gate asks:

> How much capacity does this patient need at this stage?

The off-ramp asks:

> Is the diagnosis confident enough to stop?

Neural grafting asks:

> Are some departments under-trained, and can stronger departments help them recover?

This gives us the three central mechanisms:

1. **Width routing** decides how much of a layer to use.
2. **Depth routing** decides when to stop.
3. **Neural grafting** keeps the available subnetworks healthy.

# Final Takeaway

Flexible BERT changes the compression question.

Instead of asking:

```text
How small can we make BERT?
```

it asks:

```text
How much BERT does this input need?
```

That is the deeper idea. Future efficient models should not only be smaller. They should be adaptive. They should learn when to think more, and just as importantly, when not to.
