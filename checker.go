package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/otiai10/gosseract"
)

var (
	dataset = flag.String("dataset", "", "path to dataset")
	models  = flag.String("model", "", "path to models directories")
)

func init() {
	flag.Parse()
	if *dataset == "" || *models == "" {
		flag.PrintDefaults()
		os.Exit(0)
	}
}

func main() {
	pairs, err := buildPairs(*dataset)
	if err != nil {
		log.Fatalln("pairs building failed:", err)
	}
	log.Printf("pairs building finished pairs_count=%d\n", len(pairs))

	var successful int32
	var wg sync.WaitGroup
	for k, v := range pairs {
		wg.Add(1)
		go func(target string, photos []string, wg *sync.WaitGroup) {
			defer wg.Done()

			t, err := parseTarget(target)
			if err != nil {
				log.Printf("target parsing: %v\n", err)
				return
			}
			log.Printf("target %q has been parsed", target)

			recognized := recognize(v, *models)
			if len(recognized) == 0 {
				log.Println("recognition failed, nothing to post-process.")
				return
			}

			stat := InitStat(t, recognized).
				BuildHeatMap().
				MedianRecognitionLatency().
				MedianLevenshteinDistance().
				Precision().
				Recall()

			if err := stat.Dump(target); err != nil {
				log.Printf("can't dump results of target %q: %v", target, err)
			}
			atomic.AddInt32(&successful, 1)
		}(k, v, &wg)
	}
	wg.Wait()
	log.Printf("%d targets has been processed. Check .out files.", atomic.LoadInt32(&successful))
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
			curDir := path.Join(dataset, dir.Name())

			sub, err := ioutil.ReadDir(curDir)
			if err != nil {
				log.Printf("error: can't read subdirectory %q: %v\n", curDir, err)
				return
			}

			var target string
			v := make([]string, 0)

			for _, f := range sub {
				if f.Size() == 0 || f.IsDir() {
					atomic.AddUint64(&skippedCount, 1)

					log.Printf("%q skipped (due to empty file or directory found) %v\n", curDir, err)
					continue
				}
				if strings.HasSuffix(f.Name(), ".target") {
					target = path.Join(curDir, f.Name())
					continue
				}
				v = append(v, path.Join(curDir, f.Name()))
			}

			if target == "" {
				log.Printf("%q entirely skipped: .target file was not found\n", curDir)
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
	log.Printf("targets are follows:\n")
	for name, v := range idx {
		fmt.Printf("\tpath=%q size=%d\n", name, len(v))
	}

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
func recognize(picturePaths []string, modelPath string) []recognition {
	client := gosseract.NewClient()
	client.TessdataPrefix = &modelPath
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
		text = strings.Replace(text, "\n", "", -1)
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
	Lines int    `json:"lines"`
	Text  string `json:"text"`
}

// parseTarget gets path to target file and parses it into target structure.
func parseTarget(targ string) (*target, error) {
	f, err := os.OpenFile(targ, os.O_RDONLY, 0660)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	t := new(target)
	if err := json.NewDecoder(f).Decode(t); err != nil {
		return nil, err
	}
	return t, nil
}

// statx holds all available statistics over recognitions
type statx struct {
	// initial data
	tg  *target
	rec []recognition
	// First rune on every position is expected rune. All runes after are incorrectly
	// recognized runes. position:[]recognized_rune.
	heat      map[int][]rune
	precision map[string]float64 // k:v are path_to_input:recognition_precision
	recall    map[string]float64 // k:v are path_to_input:recognition_recall
	result    map[string]string  // k:v are path_to_input:recognized_text

	totalPhotos       uint64 // total photo count
	avgLatency        uint64 // simple arithmetic average of recognition latency
	medianLatency     uint64 // median recognition latency (half latencies are strictly more, another half strictly less)
	medianLevenshtein uint64 // median recognition error rate
	groundTruth       string // expected recognition value
}

func InitStat(tg *target, rec []recognition) *statx {
	return &statx{tg: tg, rec: rec}
}

func (s *statx) BuildHeatMap() *statx {
	var (
		lock       sync.Mutex
		heat       = make(map[int][]rune)
		results    = make(map[string]string)
		avgLatency uint64
	)

	for i, r := range s.rec {
		avgLatency += uint64(r.latency)
		results[r.source] = r.result

		s.rec[i].levDist = levenshtein(s.tg.Text, r.result)

		for i := 0; i < len(s.tg.Text); i++ {
			lock.Lock()
			vec, ok := heat[i]
			if !ok {
				vec = []rune{rune(s.tg.Text[i])}
			}
			vec = append(vec, rune(r.result[i]))
			heat[i] = vec
			lock.Unlock()
		}
	}

	s.avgLatency = avgLatency / uint64(len(s.rec))
	s.heat = heat
	s.totalPhotos = uint64(len(s.rec))
	s.result = results
	s.groundTruth = s.tg.Text

	return s
}

func (s *statx) MedianRecognitionLatency() *statx {
	r := recognitionsSortLatency(s.rec) // typecast for sort by latency interface
	sort.Sort(&r)

	m := r.Len() / 2
	s.medianLatency = uint64(r[m].latency)
	if r.Len()%2 != 0 {
		s.medianLatency = (s.medianLatency + uint64(r[m+1].latency)) / 2
	}
	return s
}

func (s *statx) MedianLevenshteinDistance() *statx {
	r := recognitionsSortLevenshtein(s.rec) // typecast for sort by levDist
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
func (s *statx) Precision() *statx {
	precision := make(map[string]float64)
	for _, r := range s.rec {
		v := float64(len(s.tg.Text)-r.levDist) / float64(len(s.tg.Text))
		precision[r.source] = v
	}

	s.precision = precision
	return s
}

// Recall is TP/(TP+FN), where
//  - TP: true positive; all correctly recognized characters
//  - FN: false negative; wanted to recognize but got space or incorrect character from ~Alphabet
func (s *statx) Recall() *statx {
	recall := make(map[string]float64)


	for _, r := range s.rec {
		var (
			fn int
			tp int

			want = strings.NewReader(s.tg.Text)
			act  = strings.NewReader(r.result)
		)

		for {
			wanted, err := want.ReadByte()
			if err != nil {
				break
			}
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

		fmt.Printf("recall tp=%d fn=%d\n", tp, fn)
		recall[r.source] = float64(tp) / float64(tp+fn)
	}
	s.recall = recall
	return s
	// eval F-measure
}

func (s *statx) Dump(targetPath string) error {
	fmt.Printf("target=%q total_photos=%d avg_recognition_latency=%s "+
			"median_recognition_latency=%s median_levenshtein=%d\n",
		targetPath, s.totalPhotos, time.Duration(s.avgLatency), time.Duration(s.medianLatency), s.medianLevenshtein)
	fmt.Printf("ground truth:\n\t%v\n", s.groundTruth)
	fmt.Println("results:")
	for k, v := range s.result {
		fmt.Printf("\t%v\t\t:%v\n", v, k)
	}
	fmt.Printf("confusion heatmap of size %d:\n\twant | got\n", len(s.heat))
	for k := 0; k < len(s.heat); k++ {
		fmt.Printf("\t%v: %v\n", string(s.heat[k][0]), strings.Split(string(s.heat[k][1:]), ""))
	}
	fmt.Println("recall (more is better?):")
	for k, v := range s.recall {
		fmt.Printf("\t%v:%.3f\n", k, v)
	}

	fmt.Println("precision (more is better):")
	for k, v := range s.precision {
		fmt.Printf("\t%v: %.3f\n", k, v)
	}
	fmt.Printf("dumped_at=%s with love <3\n", time.Now())
	return nil
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
