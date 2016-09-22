# groups is a dictionary, use strings for keys, and values should be an array of strings with the columnt names
groups['Copies'] = ["MemCpy_H2C","MemCpy_C2H","TaskWaitCopy","InBufferInit","OutBufferInit","cudaMemCpy_H2G","cudaMemCpy_G2H"]
groups['Allocs'] = ["MemFree","cudaMalloc_Inputs","cudaMalloc_Outputs","cudaMemFree","TaskFree"]
groups['Kernel'] = ["Kernel Execution_CPU","Kernel_Execution_GPU"]
groups['Others'] = ["Issue","TraceIssue","Gap2Controller","TaskWaitWait","Gap2Free"]
