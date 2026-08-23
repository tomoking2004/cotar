"""The study's code: running its experiment, and reading back what that left behind.

`training` runs it, `analysis` reads it; `data`, `models`, `types` and `pairwise` are the
machinery both are built from. What is specific to *this* study — where the files sit
(`config`) and which runs are reported (`analysis.experiment`) — is kept out of that
machinery, which takes its paths as arguments and names no experiment.

Nothing is re-exported here, deliberately. `cfg` above all names one machine's directory
layout, and importing the package is not a reason to bind them: whoever needs it says so.
"""
