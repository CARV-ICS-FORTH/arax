package Vinetalk;
import com.sun.jna.Structure;

public class VineTask
{
	public VineTask(VineProcedure proc,Structure args)
	{
		this.proc = proc;
		this.args = args;
	}
	private VineProcedure proc;
	private Structure args;
}
