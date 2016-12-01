package Vinetalk;
import java.util.ArrayList;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;

public class VineTask
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
		return proc.getPointer();
	}

	public VineBuffer getArgs()
	{
		if(args == null)
			return null;
		args.write();
		return new VineBuffer(args);
	}

	public VineBuffer[] getInputs()
	{
		return (VineBuffer[])inputs.toArray(new VineBuffer[inputs.size()]);
	}

	public VineBuffer[] getOutputs()
	{
		return (VineBuffer[])outputs.toArray(new VineBuffer[outputs.size()]);
	}

	public void setArgs(Structure args)
	{
		this.args = args;
		args.write();
	}

	public void setTask(Pointer task)
	{
		this.task = task;
	}

	public void addInput(byte [] data)
	{
		inputs.add(new VineBuffer(data));
	}

	public void addOutput(byte [] data)
	{
		outputs.add(new VineBuffer(data,false));
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
			for(VineBuffer vb : outputs)
				vb.read();
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
