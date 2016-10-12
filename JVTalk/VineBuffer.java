package Vinetalk;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import java.util.List;
import java.util.Arrays;
import com.sun.jna.Memory;


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
		this(struct,true);	// Synchronize by default
	}

	public VineBuffer(Structure struct,boolean sync)
	{
		user_buffer = struct.getPointer();
		user_buffer_size = struct.size();
		if(sync)
			write();
	}

	public VineBuffer(byte [] data)
	{
		this(data,true);	// Synchronize by default
	}

	public VineBuffer(byte [] data, boolean sync)
	{
		Pointer mem = new Memory(data.length);
		mem.write(0,data,0,data.length);
		user_buffer = mem;
		user_buffer_size = data.length;
		if(sync)
			write();
	}

	public byte [] getByteArray()
	{
		read();
		byte [] data = user_buffer.getByteArray(0,(int)user_buffer_size);
		return data;
	}
}
