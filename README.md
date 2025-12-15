Mel
===

Work-in-progress tools to help identify new and changing moles on the skin with
the goal of early detection of melanoma skin cancer.

[![Tests](https://github.com/aevri/mel/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/aevri/mel/actions?query=branch%3Amain+)

The [Mayo Clinic][1] says:
> The first melanoma signs and symptoms often are:
>
> - A change in an existing mole
> - The development of a new pigmented or unusual-looking growth on your skin

The [UK National Health Service][2] says:
> The most common sign of melanoma is the appearance of a new mole or a change
> in an existing mole. This can happen anywhere on the body, but the back,
> legs, arms and face are most commonly affected.

To help you identify new and changing moles, the tools enable you to:

1. Maintain a catalog of your moles
2. Record a visual history of each mole

For the catalog to be useful, it must enable you to uniquely distinguish every
mole from the other new and pre-existing moles. To this end you will be able to
assign a name, identifying images and notes for each one.

For the visual history of each mole to be useful, it must enable you to
identify significant differences in moles over time. The tools will help you to
normalise the rotation and centering of your mole images so that they are
easier to compare.

Installation
------------

Requirements: Python 3.8+

Install uv (recommended):
```bash
pip install uv
# or
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install mel:
```bash
uv venv
uv pip install -e '.[dev]'
```

Things you can use now
----------------------

["Mel Zero"](mel_zero.md) - no executables, just documentation of how to create
and maintain a mole catalog manually using a smartphone, a laptop, and a USB
microscope.

"Mel One" - the code in this repository provides WIP rough tools for maintaining a mole catalog using two complimentary techniques. Moles are assigned UUIDs so they map across techniques:

- **Roto-mapping**: Detect new moles with the `mel rotomap *` commands. Using the idea of 'photographing a rotating cylinder and cycling through the photographs' as a cheap way of presenting an interactive 3D model, treat parts of the body as if they were a cylinder and create photosets of them. Provide support for manually and automatically marking moles in the images and mapping known ones to a UUID using some basic deep learning. Also provide support for manually comparing moles, but this better done with microscope images.

- **Microscope imaging**: Detect changes in moles with the `mel micro *` commands. These help you quickly capture and manually compare images of known moles with a USB microscope.

Other projects to watch
-----------------------

- **[Intelligent Total Body Scanner for Early Detection of Melanoma (iToBoS)](https://www.itobos.eu)**:
  - iToBoS was an EU-funded project (€11M) that developed an AI-powered full-body scanner capable of imaging a patient's entire skin surface in six minutes and providing automated risk assessments for each mole to enable earlier melanoma detection.
  - The initial four year pilot is now concluded and ran until March 2025.
  - Next steps include a large-scale roll-out trial that will involve a much larger number of melanoma cases. “While this pilot study demonstrated the technical functionality and clinical potential of the platform, a broader trial is now needed,” [notes project coordinator Rafael Garcia from the University of Girona](https://cordis.europa.eu/article/id/460600-ai-driven-diagnostic-platform-to-tackle-melanoma).

License
-------

Distributed under the Apache License (version 2.0); see the [LICENSE](LICENSE)
file at the top of the source tree for more information.

[1]: http://www.mayoclinic.org/diseases-conditions/melanoma/basics/symptoms/con-20026009
[2]: http://www.nhs.uk/Conditions/Malignant-melanoma
