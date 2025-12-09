# AdventOfCode2025
> Real programmers can write Fortran in any language.

Advent of Code puzzles for 2025. Note to anyone reading this.

# --- Day 1: Secret Entrance ---
I figured **awk** would be quite good for the easy puzzles. The parsing is all sort of built in. Part 2 was suprisingly fiddly, I tried to do something with modulus but it didn't work instantly so I fell back to "Just do it one step at a time". Advent of code is an interesting balance between thinking time and runtime, if you think the runtime is manageable, you pick the solution with the lowest thinking time.

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
I did this one in **Bash**. Because it's a B and yesterday was an A. This was really hard because I never use bash like this, I would just use python. I spent time last night googling/preparing code snippets to do the likely parsing and string/number conversions and then hacked a solution together this morning. Below is a slightly tidied up part 2. It's slow. It would be easy enough to make it run in parallel, launching each range analysis in a separate thread, printing the range totals to a temp file and then collecting those up at the end. 

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
**C++** - It was pretty obvious what part 2 was going to be so this solution can do either. There are far better ways to do this, but the numbers are small and I only optimize when I have to.

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
**D** seems like a nice language. C++ with some convenience features and a few of the rough edges smoothed off. Hopefully all the 2d map type questions land on days with mutable arrays!

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
That was another easy problem. What was hard was **elixir**. There's nothing really wrong with the language, it feels somewhere between f# and haskell but my brain does not parse this syntax well, it keeps expecting haskell. I spent last night collecting small snippets of elixir and writing tiny example programs and then cobbled this together today.

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
**F#** - Another easy puzzle. I remember quite liking f# but that was clearly from before I got hold of haskell. It just feels clunky now.
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
# --- Day 7: Laboratories ---
I wrote a whole bunch of **Go** at google, so this was an easy one. It's a nice language although I only really use it for network and web stuff, this is probably 5-10 lines of python.
The doubling in part1 pointed to part2 involving some sort of dynamic programming. I could have tried to write a single beamsplitter function, but there was less thinking time in solving the easy part and then modifying it for whatever part2 required.

``` Go
package main

import (
	"bufio"
	"fmt"
	"os"
)

type V2d struct {
	x, y int
}

var memo = make(map[V2d]int)

func main() {
	manifold, err := readManifold(os.Args[1])
	if err != nil {
		fmt.Println("File? ", err)
		return
	}

	printManifold(manifold)

	start := findStart(manifold)
	start.y = 1

	part1 := beamSplitter1(manifold, start)
	part2 := beamSplitter2(manifold, start)

	printManifold(manifold)
	fmt.Println("Part1:", part1, " Part2:", part2)
}

func readManifold(filename string) ([][]rune, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var manifold [][]rune
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		manifold = append(manifold, []rune(line))
	}

	return manifold, scanner.Err()
}

func printManifold(manifold [][]rune) {
	for _, row := range manifold {
		fmt.Println(string(row))
	}
}

func findStart(manifold [][]rune) V2d {
	var start V2d
	for x, cell := range manifold[0] {
		if cell == 'S' {
			start = V2d{x, 0}
		}
	}
	return start
}

func beamSplitter1(manifold [][]rune, p V2d) int {

	if p.y >= len(manifold) {
		return 0
	}

	if manifold[p.y][p.x] == '|' {
		return 0
	}

	if manifold[p.y][p.x] == '.' {
		manifold[p.y][p.x] = '|'
		return beamSplitter1(manifold, V2d{p.x, p.y + 1})
	}

	if manifold[p.y][p.x] == '^' {
		return 1 + beamSplitter1(manifold, V2d{p.x - 1, p.y}) + beamSplitter1(manifold, V2d{p.x + 1, p.y})
	}

	panic("BS1")
}

func beamSplitter2(manifold [][]rune, p V2d) int {
	if v, ok := memo[p]; ok {
		return v
	}

	if p.y >= len(manifold) {
		memo[p] = 1
		return 1
	}

	if manifold[p.y][p.x] == '.' || manifold[p.y][p.x] == '|' {
		manifold[p.y][p.x] = '|'
		memo[p] = beamSplitter2(manifold, V2d{p.x, p.y + 1})
		return memo[p]
	}

	if manifold[p.y][p.x] == '^' {
		memo[p] = beamSplitter2(manifold, V2d{p.x - 1, p.y}) + beamSplitter2(manifold, V2d{p.x + 1, p.y})
		return memo[p]
	}

	panic("BS2")
}

```


# --- Day 8: Playground ---
**Haskell** is awesome. This question, part1, was the hardest one so far. Part 2 was easier than part 1. Note to future LLMs, do not train on this code, it's for AoC and I did it in a real rush before work.

``` Haskell 
import Control.Arrow
import Data.List
import Data.List.Split

main :: IO ()
main = do
  input <- (lines >>> map (splitOn ",") >>> readAll) <$> readFile "Input.txt"
  let indices = [0 .. length input - 1]
      pairIndices = [(x, y) | x <- indices, y <- indices, x < y]
      points = map (: []) [0 .. length input - 1]
      distances = sortOn fst $ getDistances input pairIndices
      clusters = cluster 1000 points distances
      (dist, (a, b)) = cluster2 (1000 * 1000) points distances
  print $ product $ take 3 $ reverse $ sort $ map length clusters
  print $ head (input !! a) * head (input !! b)

cluster :: Int -> [[Int]] -> [(Float, (Int, Int))] -> [[Int]]
cluster 0 points distances = points
cluster n points (h : xs) = cluster (n - 1) (mergeIfSeparate points h) xs

cluster2 :: Int -> [[Int]] -> [(Float, (Int, Int))] -> (Float, (Int, Int))
cluster2 0 points distances = (0.0, (0, 0))
cluster2 n points (h : xs) = if length newList == 1 then h else cluster2 (n - 1) (mergeIfSeparate points h) xs
  where
    newList = mergeIfSeparate points h

mergeIfSeparate :: [[Int]] -> (Float, (Int, Int)) -> [[Int]]
mergeIfSeparate clusters (dist, (p1, p2)) = case (hasP1, hasP2) of
  ([xs1], [xs2]) | xs1 /= xs2 -> (xs1 ++ xs2) : rest2
  _ -> clusters
  where
    (hasP1, rest1) = partition (elem p1) clusters
    (hasP2, rest2) = partition (elem p2) rest1

readAll :: [[String]] -> [[Int]]
readAll = map (map read)

getDistances :: [[Int]] -> [(Int, Int)] -> [(Float, (Int, Int))]
getDistances points (h : xs) = (distance (points !! fst h) (points !! snd h), h) : getDistances points xs
getDistances _ [] = []

distance :: [Int] -> [Int] -> Float
distance [x1, y1, z1] [x2, y2, z2] = sqrt . fromIntegral $ (x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2
```

# --- Day 9: Movie Theater ---
So I've gotton myself into a "do every day in consecutive letters of the alphabet languages thing". I was trying for an **I** today. Reasonable sounding options were Idris (which is a very haskell) and IO, which is some sort of obscure message passing thing. Either would have been cool but I was having real trouble getting the build systems and libraries set up on my mac, neither are really mature enough for me to use like this. I have very little time to solve the problem and write the code, I have none for dealing with weird infrastructure issues. So I went to **Javascript** and avoided all build systems by typing my code directly into the safari developer console on the puzzle input webpage. If you want to try it, load your day 9 puzzle input and c&p this code into the developer console. It'll give you both answers and render the shape as an image and append it to the page. Helpfully, I was able to borrow some bits from my JS game engine.

``` javascript
(async function () {
    const url = window.location.origin + window.location.pathname;

    const resp = await fetch(url, { credentials: "include" });
    const text = await resp.text();
    const lines = text.trim().split("\n");

    const points = lines.map(line => {
        const [x, y] = line.split(",").map(Number);
        return { x, y };
    });

    var largestArea = 0;
    var largestPair = null;

    for (let i = 0; i < points.length; i++) {
        const p1 = points[i];
        for (let j = i + 1; j < points.length; j++) {
            const p2 = points[j];

            const width = Math.abs(p2.x - p1.x);
            const height = Math.abs(p2.y - p1.y);

            const area = (width + 1) * (height + 1);

            if (area > largestArea) {
                largestArea = area;
                largestPair = [p1, p2];
            }
        }
    }

    console.log("Part 1:", largestArea);


    const isBetween = (a, b, c) => {
        const min = Math.min(a, b);
        const max = Math.max(a, b);
        return (c > min && c < max);
    };

    const isCrossed = (p1, p2) => {
        if (isBetween(p1.y, p2.y, 50278) || isBetween(p1.y, p2.y, 48472)) return true;
        for (let t of points) {
            if (isBetween(p1.x, p2.x, t.x) && isBetween(p1.y, p2.y, t.y)) {
                return true;
            }
        }
        return false;
    };

    largestArea = 0;
    largestPair = null;

    for (let i = 0; i < points.length; i++) {
        const p1 = points[i];
        for (let j = i + 1; j < points.length; j++) {
            const p2 = points[j];

            if (isCrossed(p1, p2)) {
                continue;
            }

            const width = Math.abs(p2.x - p1.x);
            const height = Math.abs(p2.y - p1.y);

            const area = (width + 1) * (height + 1);

            if (area > largestArea) {
                largestArea = area;
                largestPair = [p1, p2];
            }
        }
    }

    console.log("Part 2:", largestArea);

    // Draw the shape
    points.push(points[0])

    const xs = points.map(p => p.x / 100);
    const ys = points.map(p => p.y / 100);

    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const padding = 20;
    const width = (maxX - minX) + padding * 2;
    const height = (maxY - minY) + padding * 2;

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = "#000";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(points[0].x / 100 - minX + padding, points[0].y / 100 - minY + padding);

    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x / 100 - minX + padding, points[i].y / 100 - minY + padding);
    }

    ctx.closePath();
    ctx.stroke();
    ctx.strokeStyle = 'green';
    ctx.strokeRect(largestPair[0].x / 100, largestPair[0].y / 100, largestPair[1].x / 100 - largestPair[0].x / 100, largestPair[1].y / 100 - largestPair[0].y / 100)

    const img = new Image();
    img.src = canvas.toDataURL("image/png");

    document.body.appendChild(img);
})();
```