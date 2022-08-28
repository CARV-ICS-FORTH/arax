package Arax;
import com.sun.jna.Pointer;

public class AraxProcedure extends AraxObject
{
	public AraxProcedure(Pointer ptr)
	{
		super(ptr);
	}
	public void release()
	{
		AraxInterface.INSTANCE.arax_proc_put(getPointer());
	}
}
