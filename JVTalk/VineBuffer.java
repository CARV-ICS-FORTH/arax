package Vinetalk;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import java.util.List;
import java.util.Arrays;

public class VineBuffer extends Structure
{
	public Pointer user_buffer;
	public long user_buffer_size;
	public Pointer vine_data;
	protected List<String> getFieldOrder()
	{
		return Arrays.asList(new String[] {"user_buffer","user_buffer_size","vine_data"});
	}

	public VineBuffer(Structure struct)
	{
		user_buffer = struct.getPointer();
		user_buffer_size = struct.size();
		write();
	}
}
