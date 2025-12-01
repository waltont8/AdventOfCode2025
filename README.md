# AdventOfCode2025
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