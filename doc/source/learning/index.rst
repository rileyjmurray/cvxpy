Getting Started
===============

TODO: route people to descriptions that go into different levels of detail.

Each description begin by answering the following

* What is optimization?
    * How are optimization models created programatically?
    * Provide three examples (expandable / collapsable).
* What is a solver?
    * Give examples of interfaces and their limitations.
* How does CVXPY help with optimization?
    * Give examples of problem reformulation.
    * Argue for the benefits of using CVXPY.

Then we can jump off into "modules"

* Basic convex optimization modeling
    * Convexity.
    * The addition, scaling, and affine-transformation rules.
    * CVXPY’s atoms
    * Mixed-integer modeling.
* Intermediate convex optimization modeling
    * Full DCP ruleset.
    * Choosing a solver.
* CVXPY-supported extensions of convex programming.
    * DGP, DQCP.
* Common painful lessons when using optimization
    * Limitations of numerical optimization: can't do better than 1e-8.
    * Infeasible and unbounded problems.