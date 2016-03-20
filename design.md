Design
======

Some more detail on the design of the system.

Useful ideas
------------

- Grouping moles into 'constellations', like stars, may make them easier to
  identify
- Splitting the body into regions for separate consideration will make it
  easier to manage large numbers of moles
- Supporting incrementally and partially updating the mole catalog and
  histories will mean that they can be updated in sessions. This may mean the
  difference between being able to do 10 minutes every day and having to spend
  a couple of hours each month for a full update.
- Simply maintaining a count of moles in a particular region may be a useful
  minimum effort for mole management
- Support for voice commands and audible feedback may make difficult to capture
  regions more managable
- Being able to filter moles and regions that require assistance to capture
  will make sessions more productive, as less skipping will be involved

Usage scenarios
---------------

1. Create a new catalog
2. Add moles to the catalog which the user already knows to be new to the
   catalog
3. Update the catalog, searching for moles on the skin that may be new to the
   catalog; without relying on the user's knowledge. Update partially, so that
   the user doesn't have to cover the whole body in one session.
4. Archive a mole from the catalog that is known to be excised, record
   pathology results
5. Query the catalog, to see if a particular mole is new
6. Capture a detailed image of a particular mole to it's history, compare with
   previous images to look for significant changes.
7. Browse the history of a particular mole, to look for significant changes
8. Capture new detailed images for the histories of moles, searching for moles
   on the skin that may have significant changes. Update partially, so that the
   user doesn't have to cover the whole body in one session.
9. Browse the detailed histories of moles, looking for significant changes that
   may have been previously overlooked

Catalog design
--------------

Important features:

- Enables determining if a particular mole is new
- Enables identifying moles missed in a previous attempt to catalog
- Easy to explore by browsing files on disk with commonly available tools

Properties to capture:

- Location on body
- Any requirement for assistance to capture
- Dated images to help identify the particular mole
- Visual history of the mole
- Any textual notes about the mole
- A short name for the mole

A simplified anatomy for approximately identifying mole locations with commonly
used terms:

- Head
- Neck
- Trunk
  - Thorax
  - Abdomen
- Arms
  - Upper arm
  - Forearm
  - Hand
- Buttocks
- Legs
  - Thigh
  - Below knee
  - Foot

See http://en.wikipedia.org/wiki/Human_body for a more detailed reference.
