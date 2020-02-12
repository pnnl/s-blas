
#include "common.h"

void matrix_transposition(const int         m,
                          const int         n,
                          const int         nnz,
                          const int        *csrRowPtr,
                          const int        *csrColIdx,
                          const VALUE_TYPE *csrVal,
                                int        *cscRowIdx,
                                int        *cscColPtr,
                                VALUE_TYPE *cscVal)
{
    // histogram in column pointer
    memset (cscColPtr, 0, sizeof(int) * (n+1));
    for (int i = 0; i < nnz; i++)
    {
        cscColPtr[csrColIdx[i]]++;
    }

    // prefix-sum scan to get the column pointer
    exclusive_scan(cscColPtr, n + 1);

    int *cscColIncr=(int *)malloc(sizeof(int) * (n+1));
    memcpy (cscColIncr, cscColPtr, sizeof(int) * (n+1));

    // insert nnz to csc
    for (int row = 0; row < m; row++)
    {
        for (int j = csrRowPtr[row]; j < csrRowPtr[row+1]; j++)
        {
            int col = csrColIdx[j];

            cscRowIdx[cscColIncr[col]] = row;
            cscVal[cscColIncr[col]] = csrVal[j];
            cscColIncr[col]++;
        }
    }

    free (cscColIncr);
}

void matrix_transposition_back(const int         n,
                          const int         m,
                          const int         nnz,
                          const int        *cscColPtr,
                          const int        *cscRowIdx,
                          const VALUE_TYPE *cscVal,
                                int        *csrColIdx,
                                int        *csrRowPtr,
                                VALUE_TYPE *csrVal)
{
    // histogram in column pointer
    memset (csrRowPtr, 0, sizeof(int) * (m+1));
    for (int i = 0; i < nnz; i++)
    {
        csrRowPtr[cscRowIdx[i]]++;
    }

    // prefix-sum scan to get the column pointer
    exclusive_scan(csrRowPtr, m + 1);

    int *csrRowIncr = (int *)malloc(sizeof(int) * (m+1));
    memcpy (csrRowIncr, csrRowPtr, sizeof(int) * (m+1));

    // insert nnz to csr
    for (int col = 0; col < n; col++)
    {
        for (int j = cscColPtr[col]; j < cscColPtr[col+1]; j++)
        {
            int row = cscRowIdx[j];

            csrColIdx[csrRowIncr[row]] = col;
            csrVal[csrRowIncr[row]] = cscVal[j];
            csrRowIncr[row]++;
        }
    }

    free (csrRowIncr);
}

