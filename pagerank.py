import os
import random
import re
import sys
import math

DAMPING = 0.85
SAMPLES = 100000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    prob = dict()
    plinks = list(corpus[page])
    pln = len(plinks)
    for link in plinks:
        prob[link] = (1/pln) * damping_factor
    links = list(corpus.keys())
    ln = len(links)
    for link in links:
        try:
            prob[link] += (1-damping_factor) * (1/ln)
        except KeyError:
            prob[link] = (1-damping_factor) * (1/ln)
    return prob

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    val = dict()
    pages = list(corpus.keys())
    page = pages[random.randint(0, len(pages)-1)]
    for r in range(n):
        prob = transition_model(corpus, page, damping_factor)
        total = 0
        for cpage in prob:
            total += prob[cpage] * 100
        total = math.floor(total)
        rand = random.randint(0, total)
        for cpage in prob:
            rand -= prob[cpage] * 100
            if rand <= 0: 
                page = cpage
                break
        try:
            val[page] += 1/n
        except KeyError:
            val[page] = 1/n
    return val


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    val = dict()
    ln = len(list(corpus.keys()))
    for page in corpus:
        val[page] = 1/ln
    while True:
        init  = val
        for cpage in corpus:
            total = 0
            links = []
            for p in corpus:
                try:
                    list(corpus[p]).index(cpage)
                    links.append(p)
                except ValueError:
                    continue
            for link in links:
                total += val[link]/len(list(corpus[link]))
            final = ((1-damping_factor)/ln) + damping_factor * total
            val[cpage] = final
        end = False
        for page in init:
            if abs(val[page] - init[page]) < 0.001:
                end = True
        if end:
            ftotal = 0
            for page in val:
                ftotal += val[page]
            if ftotal < 0.999:
                for page in val:
                    val[page] = val[page] * (1/ftotal)
            return val

if __name__ == "__main__":
    main()
