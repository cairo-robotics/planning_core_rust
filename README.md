# planning_core_rust

CAIRO Lab's forked version of the CollisionIK software developed by Daniel Rakita and colleagues at UW. 

# Cairo Lab's Motion Planning Core software built in Rust for use in constrained motion planning and per-pose optimization.

## Description and Purpose

This is a repository containts Rust source code (with a Python interface using PyO3) that provides functionality for per-pose optimization 
(RelaxedIK/CollisionIK) as well as Task Space Region / Sequential Manifold Planning pose optimization

## Table of Contents

- [Important Features](#important-features)
- [Installation](#installation)
- [FAQs](#faqs)

---
## Important Features <a name="important-features"></a>

### src/cairo_2d_sim
Location of all the source code.

#### src/

Rust source code.

src/agent/agents.rs

Contains the code for the Python interface built using PyO3. 

---
## Installation <a name="installation"></a>

Make sure you have Rust compiled onto your computer otherwise this package will fail to install. This package is built as an locally installable Python package.

In a virtual environment:

```
export SETUPTOOLS_USE_DISTUTILS=stdlib
pip3 install <path_to_this_package>
```
And in your python code:

```
from cairo_planning_core import Agent
```

Agent is the maing interface class.
 
```
self.agent = Agent(settings_fp, False, False)
```
settings_fp is the filepath location to the .yaml settings file of your designs. As an example see data/config/settings.yaml. For more information about this file refer to the RelaxedIK Core library for how this information is used.

---
## FAQ and Common Issues <a name="faqs"></a>

Please contact Carl Mueller - carl.mueller@colorado.edu for any help or if anything does not work properly.

If you see an obvious fix, want to add anything, please feel free to do a pull request.
