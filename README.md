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

# --- Day 3: Lobby ---
C++ - It was pretty obvious what part 2 was going to be so this solution can do either. There are far better ways to do this, but the numbers are small and I only optimize when I have to.

``` c++
#include <iostream>
#include <fstream>
#include <stdint.h>

const int Part1 = 2;
const int Part2 = 12;

using namespace std;

int main() {
    std::ifstream file("Input.txt");
    string row;
    uint64_t total = 0;

    int digits = Part1;

    while (file >> row) {
        uint64_t rowValue = 0;
        int max = -1;
        int start = 0;

        for (int m=0;m<digits;m++) {
            for (int n=start;n<=(row.size()-(digits-m));n++) {
                if (row[n] > max) {
                    max = row[n];
                    start = n+1;
                }
            }
            rowValue *= 10;
            rowValue = rowValue + (max - 48);
            max = -1;
        }

        total += rowValue;
    }

    cout << total << endl;

    return 0;
}
```

# --- Day 4: Printing Department ---
D seems like a nice language. C++ with some convenience features and a few of the rough edges smoothed off. Hopefully all the 2d map type questions land on days with mutable arrays!

``` D
import std.stdio;
import std.array;

int isPaper(const bool[][] grid, size_t x, size_t y) {
    if (y>=grid.length) return 0;
    if (x>=grid[y].length) return 0;
    if (y<0) return 0;
    if (x<0) return 0;
    return grid[y][x]?1:0;
}

int neighbourCount(const bool[][] grid, size_t x, size_t y) {
    int neighbours = isPaper(grid, x-1,y-1) + isPaper(grid, x,y-1) + isPaper(grid, x+1,y-1)
                   + isPaper(grid, x-1,y)                          + isPaper(grid, x+1,y)
                   + isPaper(grid, x-1,y+1) + isPaper(grid, x,y+1) + isPaper(grid, x+1,y+1);
    return neighbours;    
}

void main()
{
    bool[][] grid;
    auto lines = stdin.byLineCopy.array;

    foreach (line; lines) {
        bool[] row;
        row.reserve(line.length);

        foreach (c; line) {
            row ~= (c == '@');
        }

        grid ~= row;
    }

    // Count part 1
    int total = 0;
    foreach (y, row; grid) {
        foreach (x, cell; row) {
            if (cell) {
                int neighbours = neighbourCount(grid, x,y);
                if (neighbours < 4) {
                    total++;
                }
            }
        }
    }
    writeln(total);

    // Do part 2
    bool didChange = true;
    int removed = 0;

    while (didChange) {
        didChange = false;
        foreach (y, row; grid) {
            foreach (x, cell; row) {
                if (cell) {
                    int neighbours = neighbourCount(grid, x,y);

                    if (neighbours < 4) {
                        grid[y][x] = false;
                        didChange = true;
                        removed++;
                    }
                }
            }
        }
    }
    writeln(removed);
}
```

# --- Day 5: Cafeteria ---
That was another easy problem. What was hard was elixir. There's nothing really wrong with the language, it feels somewhere between f# and haskell but my brain does not parse this syntax well, it keeps expecting haskell. I spent last night collecting small snippets of elixir and writing tiny example programs and then cobbled this together today.

``` elixir
# Get the first argument
file = System.argv() |> List.first()

# Read in all the lines
lines = file
        |> File.read!()
        |> String.split("\n")
        |> Enum.map(&String.trim/1)

# Split on empty
[range_lines, ingredient_lines] = lines
        |> Enum.chunk_by(&(&1 == ""))
        |> Enum.reject(&(&1 == [""]))


# Get the ranges
ranges = Enum.map(range_lines, fn line ->
    [lo, hi] = line
               |> String.split("-")
               |> Enum.map(&String.to_integer/1)
    {lo, hi}
end)

# Now get the values
values = Enum.map(ingredient_lines, &String.to_integer/1)

# Count how many values fall into a range
# Watch out for overlaps
part1 = Enum.count(values, fn v ->
    Enum.any?(Enum.map(ranges, fn {lo, hi} ->
        if (v >= lo and v <= hi) , do: true, else: false
     end))
  end)


IO.puts("Part1: #{part1}")

# Lose the overlaps. Not a big surprise.
sorted = Enum.reduce(Enum.sort_by(ranges, fn {from, _to} -> from end), [], 
        fn  {from, to}, []  -> [{from, to}] 
            {from, to}, acc -> [{accFrom, accTo} | rest] = acc
                               if from <= accTo do
                                   merged = {accFrom, max(to, accTo)}
                                   [merged | rest]
                               else
                                   [{from, to} | acc]
                               end
        end)
        |> Enum.reverse()

part2 = Enum.map(sorted, fn {from, to} -> to - from + 1 end)
        |> Enum.sum()

IO.puts("Part2: #{part2}")
```

# --- Day 6: Trash Compactor ---
F# - Another easy puzzle. I remember quite liking f# but that was clearly from before I got hold of haskell. It just feels clunky now.
I tried a few recommended formatters for the code but none of them could make it look tidy. I also tried hand formatting it but it is weirdly fussy about whitespace and throws a lot of errors. Probably fine when you're used to it but I found it quite frustrating.

``` f#
open System
open System.IO

let file = Environment.GetCommandLineArgs() |> Array.skip 2 |> Array.head

let lines = File.ReadAllLines(file)

let operators = lines.[lines.Length - 1]
let numbers = lines.[0 .. lines.Length - 2]

// Strings to arrays of numbers
let rows =
    numbers
    |> Array.map (fun line ->
        line.Split([| ' '; '\t' |], System.StringSplitOptions.RemoveEmptyEntries)
        |> Array.map int64)

// rows to columns
let columns =
    Array.init rows.[0].Length (fun c -> rows |> Array.map (fun row -> row.[c]))

let ops =
    operators.Split([| ' '; '\t' |], System.StringSplitOptions.RemoveEmptyEntries)

// This is what passes for zip in f#
let results =
    Array.map2
        (fun o c ->
            if o = "*" then
                Array.reduce (fun acc x -> acc * x) c
            else
                Array.sum c)
        ops
        columns

printfn "Part1: %d" (Array.sum results)

// Transpose the array of rows and start again
// Version built into List seems to only take squares
let maxLen = numbers |> Array.map String.length |> Array.max
let transposed = 
    [| for i in 0 .. maxLen - 1 ->
           numbers
           |> Array.choose (fun s -> if i < s.Length then Some s.[i] else None)
           |> System.String |]

let part2Columns = 
    transposed
        |> Array.fold
            (fun (acc, current) line ->
                if line.Trim() = "" then
                    if current = [] then
                        acc, []
                    else
                        (List.rev current :: acc), []
                else
                    acc, (int64 (line.Trim()) :: current))
            ([], [])
        |> fun (acc, current) ->
            let allChunks = if current = [] then acc else List.rev current :: acc
            allChunks |> List.rev |> List.map List.toArray |> List.toArray

let results2 =
    Array.map2 (fun o c ->
                   if o = "*" then
                       Array.reduce (fun acc x -> acc * x) c
                   else
                       Array.sum c)
               ops
               part2Columns

printfn "Part2: %d" (Array.sum results2)

```