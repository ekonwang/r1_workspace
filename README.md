<!-- ![core_framework](images/core_visualization.png#pic_center) -->

- Pull the submodules: `git submodule init ; git submodule update --init --recursive`

- Update all submodules: `git submodule update --remote`

    - Update some submodules: `git submodule update LightZero opencompass`

    - Keep commits: `git add LightZero opencompass; git commit -m "Update submodules to match remote versions"`
