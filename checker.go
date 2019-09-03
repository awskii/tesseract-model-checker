package main

import (
	"encoding/json"
	"io/ioutil"
	"log"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/otiai10/gosseract"
)

func main() {
	if len(os.Args) < 2 {
		log.Println("data directory should be passed as second argument")
		return
	}

	pairs, err := buildPairs(os.Args[1])
	if err != nil {
		log.Fatalln("pairs building failed:", err)
	}

	var lock sync.Mutex
	stats := make([]*statx, 0)

	var wg sync.WaitGroup
	for k, v := range pairs {
		wg.Add(1)
		go func(target string, photos []string, wg *sync.WaitGroup, lock *sync.Mutex) {
			defer wg.Done()

			t, err := parseTarget(target)
			if err != nil {
				log.Printf("target parsing: %v\n", err)
				return
			}

			recognized := recognize(v)

			stat := new(statx).
				BuildHeatMap(t, recognized).
				MedianRecognitionLatency(recognized).
				MedianLevenshteinDistance(recognized).
				Precision(t, recognized).
				Recall(t, recognized)

			lock.Lock()
			stats = append(stats, stat)
			lock.Unlock()
		}(k, v, &wg, &lock)
	}

}

// buildPairs gets dataset directory and concurrently traverse next-level
// directories. One subdirectory contains one .target file and one or more
// samples to recognize such target. It's a dataset maintainer concern that
// all pictures in a subdirectory have one (correct) target.
// There are no restrictions on directory name (and they means nothing@s)
// Traversal is non-recursive.
func buildPairs(dataset string) (map[string][]string, error) {
	dirs, err := ioutil.ReadDir(dataset)
	if err != nil {
		return nil, err
	}

	var (
		lock sync.RWMutex
		wg   sync.WaitGroup

		dirCount     uint64
		filesCount   uint64
		skippedCount uint64

		idx = make(map[string][]string) // target_result:[]photo_to_recognize_name
	)

	for _, dir := range dirs {
		wg.Add(1)

		go func(wg *sync.WaitGroup, dir os.FileInfo) {
			defer wg.Done()

			if !dir.IsDir() {
				return
			}
			atomic.AddUint64(&dirCount, 1)

			sub, err := ioutil.ReadDir(dir.Name())
			if err != nil {
				log.Printf("error: can't read subdirectory %q: %v\n", dir.Name(), err)
				return
			}

			var target string
			v := make([]string, 0)

			for _, f := range sub {
				if f.Size() == 0 || f.IsDir() {
					atomic.AddUint64(&skippedCount, 1)

					log.Printf("%q skipped (due to empty or directory) %v\n", dir.Name(), err)
					continue
				}
				if strings.HasSuffix(f.Name(), ".target") {
					target = f.Name()
					continue
				}
				v = append(v, f.Name())
			}

			if target == "" {
				log.Printf("%q entirely skipped: .target file was not found\n", dir.Name())
				atomic.AddUint64(&skippedCount, uint64(len(v)))
				return
			}

			lock.Lock()
			idx[target] = v
			lock.Unlock()

			atomic.AddUint64(&filesCount, uint64(len(v)))
		}(&wg, dir)
	}
	wg.Wait()
	log.Printf("%q scanned. target_count=%d photos_count=%d skipped=%d\n", dataset, dirCount, filesCount, skippedCount)

	return idx, nil
}

type recognition struct {
	source  string
	result  string
	latency time.Duration
	levDist int // Levenshtein distance between expected and actual output
}

// validate gets paths to pictures and performs recognition.
// Evaluates recognition latency and returns slice of recognition results.
func recognize(picturePaths []string) []recognition {
	client := gosseract.NewClient()
	defer client.Close()

	recs := make([]recognition, 0, len(picturePaths))

	for _, pic := range picturePaths {
		err := client.SetImage(pic)
		if err != nil {
			log.Printf("init error: file %q: %v\n", pic, err)
			continue
		}

		s := time.Now()
		text, err := client.Text()
		if err != nil {
			log.Printf("recognition error: file %q: %v\n", pic, err)
			continue
		}
		e := time.Now().Sub(s)
		recs = append(recs, recognition{
			source:  pic,
			result:  text,
			latency: e,
		})
	}
	return recs
}

// target holds wanted recognition parameters and 100% correct result.
type target struct {
	Lines    int    `json:"lines"`
	LineSize int    `json:"line_size"`
	Text     string `json:"text"`
}

// parseTarget gets path to target file and parses it into target structure.
func parseTarget(targ string) (*target, error) {
	f, err := os.OpenFile(targ, os.O_RDONLY, 0660)
	if err != nil {
		return nil, err
	}

	t := new(target)
	if err := json.NewDecoder(f).Decode(t); err != nil {
		return nil, err
	}
	return t, nil
}

// statx holds all available statistics over recognitions
type statx struct {
	// First rune on every position is expected rune. All runes after are incorrectly
	// recognized runes. position:[]recognized_rune.
	heat      map[int][]rune
	precision map[string]float64 // k:v are path_to_input:recognition_precision
	recall    map[string]float64 // k:v are path_to_input:recognition_recall

	totalPhotos       uint64 // total photo count
	avgLatency        uint64 // simple arithmetic average of recognition latency
	medianLatency     uint64 // median recognition latency (half latencies are strictly more, another half strictly less)
	medianLevenshtein uint64 // median recognition error rate
}

func (s *statx) BuildHeatMap(tg *target, rec []recognition) *statx {
	var (
		lock sync.RWMutex
		heat = make(map[int][]rune)
		wg   sync.WaitGroup

		avgLatency uint64
	)

	for _, r := range rec {
		avgLatency += uint64(r.latency)

		wg.Add(2)
		go func(target *target, r *recognition, wg *sync.WaitGroup) {
			defer wg.Done()

			r.levDist = levenshtein(target.Text, r.result)
		}(tg, &r, &wg)

		go func(target *target, r *recognition, wg *sync.WaitGroup, heat map[int][]rune, lock *sync.RWMutex) {
			defer wg.Done()

			for i := 0; i < len(target.Text); i++ {
				if target.Text[i] == r.result[i] {
					continue
				}
				lock.RLock()
				vec, ok := heat[i]
				lock.RUnlock()
				if !ok {
					vec = make([]rune, 1)
					vec[0] = rune(target.Text[i])
				}
				vec = append(vec, rune(r.result[i]))

				lock.Lock()
				heat[i] = vec
				lock.Unlock()
			}
		}(tg, &r, &wg, heat, &lock)
	}
	wg.Wait()
	s.avgLatency = avgLatency / uint64(len(rec))
	s.heat = heat

	return s
}

func (s *statx) MedianRecognitionLatency(rec []recognition) *statx {
	r := recognitionsSortLatency(rec) // typecast for sort by latency interface
	sort.Sort(&r)

	m := r.Len() / 2
	s.medianLatency = uint64(r[m].latency)
	if r.Len()%2 != 0 {
		s.medianLatency = (s.medianLatency + uint64(r[m+1].latency)) / 2
	}
	return s
}

func (s *statx) MedianLevenshteinDistance(rec []recognition) *statx {
	r := recognitionsSortLevenshtein(rec) // typecast for sort by levDist
	sort.Sort(&r)

	m := r.Len() / 2
	s.medianLevenshtein = uint64(r[m].levDist)
	if r.Len()%2 != 0 {
		s.medianLevenshtein = (s.medianLevenshtein + uint64(r[m+1].levDist)) / 2
	}
	return s
}

// Precision is TP/(TP+FP), where
//  - TP: true positive; all correctly recognized characters
//  - FP: false positive; incorrectly recognized characters (from same alphabet!)
func (s *statx) Precision(tg *target, rec []recognition) *statx {
	precision := make(map[string]float64)
	for _, r := range rec {
		precision[r.source] = float64(len(tg.Text)-r.levDist) / float64(len(tg.Text))
	}

	s.precision = precision
	return s
}

// Recall is TP/(TP+FN), where
//  - TP: true positive; all correctly recognized characters
//  - FN: false negative; wanted to recognize but got space or incorrect character from ~Alphabet
func (s *statx) Recall(tg *target, rec []recognition) *statx {
	recall := make(map[string]float64)

	for _, r := range rec {
		var (
			fn int
			tp int

			want = strings.NewReader(tg.Text)
			act  = strings.NewReader(r.result)
		)

		for wanted, err := want.ReadByte(); err != nil; {
			actual, err := act.ReadByte()
			if err != nil {
				break
			}
			if actual == wanted {
				tp++
				continue
			}
			// if got char not from [0-9A-Z<] alphabet
			if (actual < '0' || actual > '9') && (actual < 'A' || actual > 'Z') && actual != '<' {
				fn++
			}
		}

		recall[r.source] = float64(tp) / float64(tp+fn)
	}
	s.recall = recall
	return s
	// eval F-measure
}

// Levenshtein distance between s and t.
func levenshtein(s, t string) int {
	if len(s) == 0 {
		return len(t)
	}

	if len(t) == 0 {
		return len(s)
	}

	dists := make([][]int, len(s)+1)
	for i := range dists {
		dists[i] = make([]int, len(t)+1)
		dists[i][0] = i
	}

	for j := range t {
		dists[0][j] = j
	}

	for i, sc := range s {
		for j, tc := range t {
			if sc == tc {
				dists[i+1][j+1] = dists[i][j]
			} else {
				dists[i+1][j+1] = dists[i][j] + 1
				if dists[i+1][j] < dists[i+1][j+1] {
					dists[i+1][j+1] = dists[i+1][j] + 1
				}
				if dists[i][j+1] < dists[i+1][j+1] {
					dists[i+1][j+1] = dists[i][j+1] + 1
				}
			}
		}
	}

	return dists[len(s)][len(t)]
}

// useful type wrapper for timsort over latency
type recognitionsSortLatency []recognition

func (r *recognitionsSortLatency) Len() int {
	return len(*r)
}

func (r *recognitionsSortLatency) Less(i, j int) bool {
	if (*r)[i].latency < (*r)[j].latency {
		return true
	}
	return false
}

func (r *recognitionsSortLatency) Swap(i, j int) {
	(*r)[i], (*r)[j] = (*r)[j], (*r)[i]
}

// useful type wrapper for timsort over levenshtein distance
type recognitionsSortLevenshtein []recognition

func (r *recognitionsSortLevenshtein) Len() int {
	return len(*r)
}

func (r *recognitionsSortLevenshtein) Less(i, j int) bool {
	if (*r)[i].levDist < (*r)[j].levDist {
		return true
	}
	return false
}

func (r *recognitionsSortLevenshtein) Swap(i, j int) {
	(*r)[i], (*r)[j] = (*r)[j], (*r)[i]
}
