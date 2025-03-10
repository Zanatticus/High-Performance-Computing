# Problem 5 (10 Points)

Find a published paper from an **ACM** or **IEEE** conference that discusses a novel sparse matrix format that was not covered in class. Discuss why the proposed format is superior to the **CSR** or **CSC** format. Make sure to cite your sources.

(Optional for undergraduate/Plus-One students, required for MS/PhD students. For undergraduate/Plus-One students, completing can add up to **20 points** to your quiz grade.)

## Answers

The sparse matrix formats covered in class include *CSR*, *CSC*, *COO*, and *ELLPACK*. The new format I chose to to discuss is the **diagonal format (DIA)**.

#### Diagonal Format (DIA)

The diagonal format is uniquely used for a type of sparse matrix called a band matrix. Band matrices essentially have nonzero entries appear uniformly in a diagonal band. Specifically, DIA excels compared to CSR or CSC when these band matrices exhibit a **strong** diagonal band (few nonzero elements outside of the diagonal) and with few diagonals compared to the total number of rows and columns.

Although not general purpose, this format efficiently encodes matrices arising from the application of stencils to regular grids, a common discretization method. The diagonal format is formed by two arrays: `data`, which stores the nonzero values, and `offsets`, which stores the offset of each diagonal from the main diagonal. Diagonals above and below the main diagonal have positive and negative offsets [1]. As such it means that the DIA format allows for more efficient algorithms to multiply two band matrices on the CPU and also on the GPU compared to CSR and CSC [2].

Also, this sparse matrix format is easily parallelized for SIMD execution: 
- Assign one thread to each row and store `data` in column-major order so that consecutive elements within each diagonal are adjacent. As each thread iterates over the diagonals crossing its row, this memory layout guarantees that contiguous threads access contiguous elements of the `data` array [1].
- Compared to CSR which might suffer from load imbalance due to different rows varying in nonzero elements, DIA allows for a more balanced workload distribution across threads since all rows contain the same number of elements (padding is used when necessary).

Other things to note are that DIA has a lower storage overhead compared to CSR or CSC as it only stores the nonzero elements and their offsets whereas CSR and CSC need to store the row/column indices of each nonzero element. These indices lead to higher memory usage since fetching of indeces is required first before fetching the actual data.

#### Sources Cited:
Paper that introduced common sparse matrix formats which briefly includes the diagonal format:
- [1] https://dl.acm.org/doi/pdf/10.1145/1654059.1654078

Paper that discusses the diagonal format (DIA) in more depth and its use for band matrices:
- [2] https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6507483
