# AdventOfCode2025
> Real programmers can write Fortran in any language.

Advent of Code puzzles for 2025

# --- Day 1: Secret Entrance ---
I figured awk would be quite good for the easy puzzles. The parsing is all sort of built in. Part 2 was suprisingly fiddly, I tried to do something with modulus but it didn't work instantly so I fell back to "Just do it one step at a time". Advent of code is an interesting balance between thinking time and runtime, if you think the runtime is manageable, you pick the solution with the lowest thinking time.

``` awk
#!/usr/bin/awk -f

BEGIN {
        position = 50
        part1 = 0
        part2 = 0
      }
      {        
        letter = substr($0, 1, 1)
        number = substr($0, 2) + 0 
   
        if (letter ~ "L") {
            while (number != 0) {
                position--;
                number--;
                if (position == -1) {
                    position = 99
                }
                if (position == 0) {
                    part2++
                    if (number == 0) {
                        part1++
                    }
                }
            }
        } else if (letter ~ "R") {
            while (number != 0) {
                position++;
                number--;
                if (position == 100) {
                    position = 0
                }
                if (position == 0) {
                    part2++
                    if (number == 0) {
                        part1++
                    }
                }
            }
        }
      }
END   {
        print part1,part2
      }
```

# --- Day 2: Gift Shop ---
I did this one in Bash. Because it's a B and yesterday was an A. This was really hard because I never use bash like this, I would just use python. I spent time last night googling/preparing code snippets to do the likely parsing and string/number conversions and then hacked a solution together this morning. Below is a slightly tidied up part 2. It's slow. It would be easy enough to make it run in parallel, launching each range analysis in a separate thread, printing the range totals to a temp file and then collecting those up at the end. 

``` bash
#!/bin/bash

input=$(cat "Input.txt")

total=0

repeatedN() {
    local n="$1"
    local s="$2"

    local partLength=$((len / n))
    local firstPart=${s:0:partLength}
    local i offset part

    for ((i=1; i<n; i++)); do
        offset=$((i * partLength))
        part=${s:offset:partLength}
        if [[ "$part" != "$firstPart" ]]; then
            return 1
        fi
    done

    return 0
}

repeatedAny() {
    local s="$1"
    local len=${#s}
    local n

    for ((n=2; n<=len; n++)); do
        if (( len % n == 0 )); then
            if repeatedN "$n" "$s"; then
                return 0
            fi
        fi
    done

    return 1
}

IFS=',' read -ra ranges <<< "$input"

for r in "${ranges[@]}"; do
    start=${r%-*}
    end=${r#*-}

    for ((i=start; i<=end; i++)); do
        if repeatedAny "$i"; then
            total=$((total + i))
        fi
    done
done

echo "Part 2: $total"

```