TODO list
=========

A simple list to capture outstanding tasks, roughly in priority order.

- micro add: option to disable rotation requirement
- micro add: option for bigger central box
- micro add: option to disable central box
- micro add: save selected options as files on disk e.g. '__big_box__'

- mel.lib.common: move new_image() to mel.lib.image
- mel.lib.common: move other image functions to mel.lib.image

- Rotomaps: handle the case where we've captured something in the past that
  turns out not to be a mole, e.g. it was just a spot and eventually
  disappeared. Since 'mel status' will nag us about it being a 'missing mole'
  in every new rotomap, we want to resolve it somehow. One way would be to
  delete it, that seems to throw away the useful information that it was once a
  thing though. Usually source control can be relied upon to preserve that sort
  of information. Unfortunately 'source control' and 'lots of images' isn't
  always straight-forward.
