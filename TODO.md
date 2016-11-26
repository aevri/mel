TODO list
=========

A simple list to capture outstanding tasks, roughly in priority order.

- micro-add: option to disable rotation requirement
- micro-add: option for bigger central box
- micro-add: option to disable central box
- micro-add: save selected options as files on disk e.g. '__big_box__'

- mel.lib.common: move new_image() to mel.lib.image
- mel.lib.common: move other image functions to mel.lib.image

- rotomap-edit: guess translation of mole in 'follow, paste mole' from last
  translation, if no other information. Useful for marking the first moles in a
  new set of images.
- rotomap-edit: mode indicator, i.e. is this 'follow mode' or 'follow, paste
  mode?'.
- rotomap-edit: guess position in 'follow, paste mode' of mole by optical cues
  only, for marking the first moles in a new set of images.

- Consider mole detector for marking mole positions.
- Consider mole analyser for guessing which mole is which, e.g. 'light',
  'dark', 'small', 'large', 'circular', 'elliptical'.
- Consider automatically detecting skin regions and estimating curvature, to
  factor into position estimation.
- Consider LK tracker for copying moles from one image to another without
  pre-marked moles in the destination.

- Re-arrange commands to something like this:
    - mel micro  / mel m
        - mel m add
        - mel m addcluster
        - mel m addsingle
        - mel m compare
        - mel m list
        - mel m view
    - mel rotomap  / mel r
        - mel r diff
        - mel r edit
        - mel r list
        - mel r molepicker
        - mel r organise
        - mel r overview
        - mel r relate
        - mel r show
        - mel r uuid
