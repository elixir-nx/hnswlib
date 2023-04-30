defmodule HNSWLib.Helper do
  @moduledoc false

  def get_keyword(opts, key, type, default, allow_nil? \\ false) do
    val = opts[key] || default

    if allow_nil? and val == nil do
      {:ok, val}
    else
      get_keyword(key, opts[key] || default, type)
    end
  end

  defp get_keyword(_key, val, :non_neg_integer) when is_integer(val) and val >= 0 do
    {:ok, val}
  end

  defp get_keyword(key, val, :non_neg_integer) do
    {:error,
     "expect keyword parameter `#{inspect(key)}` to be a non-negative integer, got `#{inspect(val)}`"}
  end

  defp get_keyword(_key, val, :pos_integer) when is_integer(val) and val > 0 do
    {:ok, val}
  end

  defp get_keyword(key, val, :pos_integer) do
    {:error,
     "expect keyword parameter `#{inspect(key)}` to be a positive integer, got `#{inspect(val)}`"}
  end

  defp get_keyword(_key, val, :integer) when is_integer(val) do
    {:ok, val}
  end

  defp get_keyword(key, val, :integer) do
    {:error, "expect keyword parameter `#{inspect(key)}` to be an integer, got `#{inspect(val)}`"}
  end

  defp get_keyword(_key, val, :boolean) when is_boolean(val) do
    {:ok, val}
  end

  defp get_keyword(key, val, :boolean) do
    {:error, "expect keyword parameter `#{inspect(key)}` to be a boolean, got `#{inspect(val)}`"}
  end

  defp get_keyword(_key, val, :function) when is_function(val) do
    {:ok, val}
  end

  defp get_keyword(key, val, :function) do
    {:error, "expect keyword parameter `#{inspect(key)}` to be a function, got `#{inspect(val)}`"}
  end

  defp get_keyword(_key, val, {:function, arity})
       when is_integer(arity) and arity >= 0 and is_function(val, arity) do
    {:ok, val}
  end

  defp get_keyword(key, val, {:function, arity}) when is_integer(arity) and arity >= 0 do
    {:error,
     "expect keyword parameter `#{inspect(key)}` to be a function that can be applied with #{arity} number of arguments , got `#{inspect(val)}`"}
  end

  def list_of_binary(data) when is_list(data) do
    count = Enum.count(data)

    if count > 0 do
      first = Enum.at(data, 0)

      if is_binary(first) do
        expected_size = byte_size(first)

        if rem(expected_size, HNSWLib.Nif.float_size()) != 0 do
          {:error,
           "vector feature size should be a multiple of #{HNSWLib.Nif.float_size()} (sizeof(float))"}
        else
          features = trunc(expected_size / HNSWLib.Nif.float_size())

          if list_of_binary(data, expected_size) == false do
            {:error, "all vectors in the input list should have the same size"}
          else
            {:ok, {count, features}}
          end
        end
      end
    else
      {:ok, {0, 0}}
    end
  end

  defp list_of_binary([elem | rest], expected_size) when is_binary(elem) do
    if byte_size(elem) == expected_size do
      list_of_binary(rest, expected_size)
    else
      false
    end
  end

  defp list_of_binary([], expected_size) do
    expected_size
  end
end
