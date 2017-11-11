function toggleSlice(btn,slice)
{
	elem = document.getElementById(slice);
	if(elem.className == 'slice0')
	{
		elem.className = 'slice1';
		btn.className = 'btn1';
	}
	else
	{
		elem.className = 'slice0';
		btn.className = 'btn0';
	}
}

function highlight_same(obj)
{
	if(this.prev != null)
	{
		var names = document.getElementsByName(this.prev);
		for(i = 0 ; i < names.length; i++)
		{
			names[i].className = '';
		}
	}
	if(obj.getAttribute('name') != null)
	{
		var names = document.getElementsByName(obj.getAttribute('name'));
		for(i = 0 ; i < names.length; i++)
		{
			if(names.length > 1)
				names[i].className = 'GoodBG';
			else
				names[i].className = 'BadBG';
		}
		this.prev = obj.getAttribute('name');
	}
}

function blockTogle(name)
{
	block = document.getElementsByName(name)[0];
	if(block.className == "block")
		block.className = "block_show";
	else
		block.className = "block";
}
