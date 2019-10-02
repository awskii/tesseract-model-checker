package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/awskii/gosseract"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Println("Usage: mm tessdata-prefix path-to-image")
		os.Exit(1)
	}

	c := gosseract.NewClient()
	if c == nil {
		fmt.Println("gosseract init failed")
		os.Exit(1)
	}

	c.SetTessdataPrefix(os.Args[1])
	langs, err := listLanguages(os.Args[1])
	if err != nil {
		fmt.Printf("get_lang failed: %v\n", err)
	}
	fmt.Printf("N_langs=%d; %+v\n", len(langs), langs)

	err = c.SetLanguage(langs...)
	if err != nil {
		fmt.Printf("set_langs failed: %v\n", err)
	}

	c.SetImage(os.Args[2])
	s := time.Now()
	out, err := c.Text()
	if err != nil {
		fmt.Printf("ocr (%q) failed: %v\n", os.Args[2], err)
	}
	fmt.Printf("ETA=%v result:\n%q\n", time.Now().Sub(s), out)
}

func listLanguages(dir string) ([]string, error) {
	languages, err := filepath.Glob(filepath.Join(dir, "*.traineddata"))
	if err != nil {
		return nil, err
	}
	for i := 0; i < len(languages); i++ {
		languages[i] = strings.TrimSuffix(filepath.Base(languages[i]), ".traineddata")
	}
	return languages, nil
}
