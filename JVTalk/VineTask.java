package Vinetalk;

import java.util.ArrayList;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import java.io.Serializable;

public class VineTask implements Serializable
{
    public enum State {
        Failed(0), /**< Task execution failed. */
        Issued(1), /**< Task has been issued. */
        Completed(2); /**< Task has been completed. */
        private final int value;

        State(int value)
        {this.value = value;}
        int getAsInt()
        {return value;}
    }

	public VineTask(VineProcedure proc,Structure args)
	{
		this.proc = proc;
		this.args = args;
		this.inputs = new ArrayList<VineBuffer>();
		this.outputs = new ArrayList<VineBuffer>();
	}

	public VineTask(VineProcedure proc)
	{
		this.proc = proc;
		this.inputs = new ArrayList<VineBuffer>();
		this.outputs = new ArrayList<VineBuffer>();
	}

	public Pointer getProcedure()
	{
		if( proc != null)
			return proc.getPointer();
		else
			return null;
	}

	public Pointer getArg()
	{
		if(args == null)
			return null;
		return args.getPointer();
	}

	public int getArgSize()
	{
		if(args == null)
			return 0;
		return args.size();
	}

	static private Pointer[] ioCast(ArrayList<VineBuffer> io)
	{
		if(io.size() == 0)
			return null;

		Pointer [] ret = new Pointer[io.size()];

		for(int c = 0 ; c < io.size() ; c++)
			ret[c] = io.get(c).getPointer();

		return ret;
	}

	public Pointer[] getInputs()
	{
		return ioCast(inputs);
	}

	public Pointer[] getOutputs()
	{
		return ioCast(outputs);
	}

	public VineTask setArgs(Structure args)
	{
		this.args = args;
		args.write();
		return this;
	}

	public void setTask(Pointer task)
	{
		this.task = task;
	}

	public VineTask addInput(VineBuffer vb)
	{
		inputs.add(vb);
		return this;
	}

	public VineTask addOutput(VineBuffer vb)
	{
		outputs.add(vb);
		return this;
	}

	public State status()
	{
		return status(true);
	}

	public State status(boolean sync)
	{
		int ret;

		if(sync)
			ret = VineTalkInterface.INSTANCE.vine_task_wait(task);
		else
			ret = VineTalkInterface.INSTANCE.vine_task_stat(task,null);

		if(ret == 2) // Complete
		{
			if(!sync)       // Call wait since it will not block
				VineTalkInterface.INSTANCE.vine_task_wait(task);

			VineTalkInterface.INSTANCE.vine_task_free(task);
		}
		return State.values()[ret];
	}

	private VineProcedure proc;
	private Structure args;
	private Pointer task;
	public ArrayList<VineBuffer> inputs;
	public ArrayList<VineBuffer> outputs;
}
