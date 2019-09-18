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
	dataset    = flag.String("dataset", "", "path to dataset")
	modelsPath = flag.String("model", "", "path to models directories")
	output     = flag.String("output", "", "path to put results in")
)

func init() {
	flag.Parse()
	if *dataset == "" || *modelsPath == "" {
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
	models, err := parseModelPath(*modelsPath)
	if err != nil {
		log.Fatalln("model finding failed:", err)
	}

	var successful int32
	var wg sync.WaitGroup

	type variant struct {
		target *target
		result map[string]string // k:v are model:result
	}

	var varMu sync.Mutex
	variants := make(map[string]variant) // k:v is source:variants

	for target, photos := range pairs {
		for _, model := range models {
			wg.Add(1)
			go func(target, model string, photos []string, wg *sync.WaitGroup) {
				defer wg.Done()

				t, err := parseTarget(target)
				if err != nil {
					log.Printf("target parsing: %v\n", err)
					return
				}
				log.Printf("target %q has been parsed", target)

				recognized := recognize(photos, model)
				if len(recognized) == 0 {
					log.Println("recognition failed, nothing to post-process.")
					return
				}

				stat := InitStat(t, recognized, model).
					BuildHeatMap()
				// MedianRecognitionLatency().
				// MedianLevenshteinDistance().
				// Precision().
				// Recall()

				// collect varaints for further multi-value consensus
				varMu.Lock()
				for file, res := range stat.result {
					v, ok := variants[file]
					if !ok {
						v = variant{result: make(map[string]string), target: stat.tg}
					}
					v.result[model] = res
					variants[file] = v
				}
				varMu.Unlock()

				// if err := stat.Dump(target, *output); err != nil {
				// 	log.Printf("can't dump results of target %q: %v", target, err)
				// }
				atomic.AddInt32(&successful, 1)
			}(target, model, photos, &wg)
		}
	}

	wg.Wait()

	// that govnokod employs multi-model consensus
	for file, variants := range variants {
		mv := NewMultiValue(variants.result)
		mv.source = file

		start := time.Now()
		mvr := mv.Decide()
		end := time.Now().Sub(start)

		var rec [1]recognition
		rec[0].source = file
		rec[0].result = mvr

		s := InitStat(variants.target, rec[:], "multi-model"+path.Base(file)).
			BuildHeatMap().Precision().Recall()

		var recall, precision float64
		for _, v:= range s.recall {
			recall   = v
			break
		}
		for _, v:= range s.precision {
			precision   = v
			break
		}
		fmt.Printf("recall=%.3f precision=%.3f\n", recall, precision)
		fmt.Printf("multi-value consensus built for %s\n", end)
	} // end of consensus govnokod

	log.Printf("%d targets has been processed. Check output directory.", atomic.LoadInt32(&successful))
}

func mapToVec(m map[string]string) []string {
	vec := make([]string, 0, len(m))
	for _, v := range m {
		vec = append(vec, v)
	}
	return vec
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

func parseModelPath(mpath string) ([]string, error) {
	models, err := ioutil.ReadDir(mpath)
	if err != nil {
		return nil, err
	}

	var list []string
	for _, model := range models {
		if model.Size() == 0 || model.IsDir() {
			log.Printf("%q model skipped (is empty or subdir)", model.Name())
			continue
		}
		if strings.HasSuffix(model.Name(), ".traineddata") {
			list = append(list, path.Join(mpath, model.Name()))
			continue
		}
	}

	log.Printf("model_dir=%q model_count=%d models are follows:\n", mpath, len(list))
	for _, name := range list {
		fmt.Printf("\tpath=%q\n", name)
	}
	return list, nil
}

type recognition struct {
	source  string
	result  string
	latency time.Duration
	levDist int // Levenshtein distance between expected and actual output
}

func modelName(fpath string) string {
	return strings.TrimSuffix(path.Base(fpath), ".traineddata")
}

// validate gets paths to pictures and performs recognition.
// Evaluates recognition latency and returns slice of recognition results.
func recognize(picturePaths []string, modelPath string) []recognition {
	client := gosseract.NewClient()
	pref := path.Dir(modelPath)
	client.TessdataPrefix = &pref
	client.Languages = []string{modelName(modelPath)}
	client.SetWhitelist("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<")
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
	modelPath         string // path to OCR model
}

func InitStat(tg *target, rec []recognition, model string) *statx {
	return &statx{tg: tg, rec: rec, modelPath: model}
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

		for i := 0; i < len(s.tg.Text) && i < len(r.result); i++ {
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
			if wanted == '<' {
				fn++
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

func (s *statx) Dump(targetPath, outputPath string) error {
	p := path.Join(outputPath, path.Base(s.modelPath))
	err := os.Mkdir(p, 0770)
	if err != nil {
		log.Printf("can't create model directory %q: %v", p, err)
	}
	out, err := os.Create(path.Join(p, path.Base(targetPath)+".out"))
	if err != nil {
		log.Printf("can't open output file: %v", err)
		out = os.Stdout
	}
	if out != os.Stdout {
		defer out.Close()
	}

	fmt.Fprintf(out, "[target=%q][model=%q][dumped_at=%s] with love <3\n", targetPath, s.modelPath, time.Now())
	fmt.Fprintf(out, "total_photos=%d avg_recognition_latency=%s median_recognition_latency=%s median_levenshtein=%d\n",
		s.totalPhotos, time.Duration(s.avgLatency), time.Duration(s.medianLatency), s.medianLevenshtein)
	fmt.Fprintln(out, "FILE\t\tRECALL\tPRECISION\tRESULTS")
	for k, v := range s.result {
		fmt.Fprintf(out, "%v\t%.3v\t%.3v\t\t%v\n", k, s.recall[k], s.precision[k], v)
	}
	fmt.Fprintf(out, "GROUND TRUTH:\t\t\t\t%v\n\n", s.groundTruth)
	fmt.Fprintf(out, "CONFUSION MATRIX\nwant got\n")
	for k := 0; k < len(s.heat); k++ {
		fmt.Fprintf(out, "%3v  %v\n", string(s.heat[k][0]), strings.Split(string(s.heat[k][1:]), ""))
	}
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

type MultiValue struct {
	source  string   // source image
	model   string   // model which was used for recognition
	outputs map[string]string // list of model outputs
	result  string   // final result value. Empty before Decide call.
}

func NewMultiValue(variants map[string]string) *MultiValue {
	return &MultiValue{outputs: variants}
}

func (m *MultiValue) Decide() string {
	fmt.Printf("source=%q deciding on\n", m.source)
	// build variants ('consensus vector')
	aux := make([]map[rune]int, 0, 90) // position:[recognized_rune:approved_count]
	for usedModel, mres := range m.outputs {
		mres = strings.Replace(mres, " ", "", -1)
		fmt.Printf("\t%s - %s\n", mres, usedModel)
		for i, c := range mres {
			if i >= len(aux) { // bounds check(sic!)
				v := make(map[rune]int)
				aux = append(aux, v)
			}
			aux[i][c]++
		}
	}
	// decide which rune has more than half voices
	var fin string
	half := len(m.outputs) / 2
	// fmt.Printf("half is %d\tNotion: N-normal G-garbage F-fallback V-verified S-skipped\n", half)
	for pos, runes := range aux {
		_ = pos
		var d rune       // decided rune
		var deciders int // deciders count to prevent decide on garbage (garbage has not enough deciders).
		for char, count := range runes {
			deciders += count
			// fmt.Printf("[N] pos=%2d vec=%+v ", pos, runes)
			if count > half {
				d = char
				// fmt.Printf("[V|%s]\n", string(char))
				break
			}
			// fmt.Printf("[S|%s]\n", string(char))
		}
		if deciders < (half+1) && d == rune(0) {
			// fmt.Printf("[G] pos=%2d vec=%+v [S]\n", pos, runes)
			continue
		}
		if d == rune(0) {
			// fmt.Printf("[F] pos=%2d d=%v vec=%+v\n", pos, d, runes)
			d = '?'
			// some fallback over confusion matrices, pattern knowledge and other.
		}
		fin += string(d)
	}
	m.result = fin
	fmt.Printf("decided value is %q, len=%d\n", m.result, len(m.result))
	return m.result
}
