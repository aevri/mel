Example invocations of the 'mel' command
========================================

Provide some usage examples to get people started quickly.

### Capture a new image of the least-recently-updated mole

Compares against the oldest image taken of the same mole.

```
$ mel list --sort lastmicro | head -1 | xargs mel micro-add
```

### Capture a new image of the least-recently-updated mole, compare recent

**Warning**: '--min-compare-age-days' introduces the additional risk of
ignoring potentially significant changes.

Capture a new image and try to display the newest comparison image that is at
least a certain number of days old. The aim here is to try to ignore changes
that have happened over a longer period of time.

**Disclaimer**: You should make your own informed decision about what a
sensible comparison period is, the author is not a medical professional and
does not offer medical advice.

Omitting the '--min-compare-age-days' parameter will compare against the oldest
image and does not carry the same risk of ignoring potentially significant
changes over a longer period.

'365' is what the author uses personally as the comparison interval here,
because of [this sentence from the NHS][1]:

> See your GP as soon as possible if you notice changes in a mole, freckle or
> patch of skin, especially if the changes happen over a few weeks or months.

Setting the compare age to '365 days' means that:
- If you take new images every day, you'll still see changes that are apparent
  over a period of 365 days.
- If you take new images only every 2 years (this is probably not often enough)
  then you'll still see changes over a 2 year period.
- Caution: if changes are only visible over a 2 year period, instead of one
  year, then it is possible that the changes will not be noticed.

```
$ mel list --sort lastmicro | head -1 | xargs mel micro-add --min-compare-age-days 365
```

### Edit the '__changed__' file for the most recently updated mole

If you spot differences, it's possibly a good idea to keep track of what
appears to be different. If you're managing the files with a version tracker
like 'git' then you'll be able to keep a useful history.

```
bash$ mel list --sort lastmicro | tail -1
bash$ $EDITOR $(!!)/__changed__
```

### Print an overview of the age of the last microscope images

Quickly get an idea of when you last took a microscope image of a mole by
outputting the moles in columns, sorted by the last capture date.

```
$ mel list --format '{lastmicro_age_days} {relpath}' | sort -n | column -t | column
```

Example output:
```
0    RightLeg/Upper/KneeTriangle/TinyDark        29   RightLeg/LowerLeg/Shin
0    RightLeg/Upper/NearTSpecks/InnerHigher      29   RightLeg/Upper/Outside
26   LeftLeg/UpperLeg/LeftSidePair/Lighter
26   LeftLeg/UpperLeg/RightSidePair/LongThin
26   LeftLeg/UpperLeg/RightSidePair/SmallDark
```

[1]: http://www.nhs.uk/Conditions/Malignant-melanoma/Pages/Symptoms.aspx
