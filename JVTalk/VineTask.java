package Vinetalk;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;

public class VineTask
{
	public VineTask(VineProcedure proc,Structure args)
	{
		this.proc = proc;
		this.args = args;
	}
	public Pointer getProcedure()
	{
		return proc.getPointer();
	}
	public VineBuffer getArgs()
	{
		args.write();
		return new VineBuffer(args);
	}
	private VineProcedure proc;
	private Structure args;
}
