package Vinetalk;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import java.util.List;
import java.util.Arrays;
import com.sun.jna.Memory;
import com.sun.jna.Native;

public class VineBuffer extends Structure
{
	public Pointer user_buffer;
	public long user_buffer_size;
	public Pointer vine_data;
	private Object juser_buffer;
	private Class juser_class;

	protected List<String> getFieldOrder()
	{
		return Arrays.asList(new String[] {"user_buffer","user_buffer_size","vine_data"});
	}

	public VineBuffer()
	{}

	public VineBuffer(Structure struct)
	{
		this(struct,true);	// Synchronize by default
	}

	public VineBuffer(Structure struct,boolean sync)
	{
		user_buffer = struct.getPointer();
		user_buffer_size = struct.size();
		juser_buffer = null;
		if(sync)
			write();
	}

	public VineBuffer(float [] data)
	{
		this(data,true);	// Synchronize by default
	}

	public VineBuffer(float [] data, boolean sync)
	{
		Pointer mem = new Memory(data.length*Native.getNativeSize(Float.class));
		for(int c = 0 ; c < data.length ; c++)
			mem.setFloat(c,data[c]);
		user_buffer = mem;
		user_buffer_size = data.length;
		juser_buffer = data;
		juser_class = Float.class;
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
		juser_buffer = data;
		juser_class = Byte.class;
		if(sync)
			write();
	}

	public void copyFrom(VineBuffer source)
	{
				user_buffer = source.user_buffer;
				user_buffer_size = source.user_buffer_size;
				vine_data = source.vine_data;
				juser_buffer = source.juser_buffer;
	}

	public void read()
	{
		super.read();
		if(user_buffer == null)
			return;
		if(juser_class == Byte.class)
		{
			byte [] data = user_buffer.getByteArray(0,(int)user_buffer_size);
			if(juser_buffer != null)
				System.arraycopy(data,0,juser_buffer,0,(int)user_buffer_size);
		}
		else
		if(juser_class == Float.class)
		{
			float [] data = user_buffer.getFloatArray(0,(int)user_buffer_size);
			if(juser_buffer != null)
				System.arraycopy(data,0,juser_buffer,0,(int)user_buffer_size);
		}
		assert null=="Invalid juser_class!";
	}
}
